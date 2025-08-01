#include "frame_interpolator.h"
#include "vk_layer_frame_interpolation.h"
#include <fstream>
#include <vector>
#include <cstring>
#include <iostream>

// SPIR-V loading helper
static std::vector<uint32_t> loadSPIRV(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open SPIR-V file: " + filename);
    }
    
    size_t file_size = (size_t)file.tellg();
    std::vector<uint32_t> code(file_size / sizeof(uint32_t));
    
    file.seekg(0);
    file.read((char*)code.data(), file_size);
    file.close();
    
    return code;
}

FrameInterpolator::FrameInterpolator(DeviceData* device_data) 
    : device_data_(device_data) {
    
    createConstantsBuffer();
    createShaderModules();
    createPipelines();
    initializeFidelityFX();
    
    // Start compute thread
    compute_thread_ = std::thread(&FrameInterpolator::computeThreadFunc, this);
}

FrameInterpolator::~FrameInterpolator() {
    // Stop compute thread
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        should_stop_ = true;
    }
    queue_cv_.notify_all();
    if (compute_thread_.joinable()) {
        compute_thread_.join();
    }
    
    // Wait for device idle
    device_data_->dispatch_table.DeviceWaitIdle(device_data_->device);
    
    // Cleanup all swapchain resources
    for (auto& [swapchain, resources] : swapchain_resources_) {
        cleanupSwapchain(swapchain);
    }
    
    shutdownFidelityFX();
    
    // Destroy pipelines
    auto& dt = device_data_->dispatch_table;
    VkDevice device = device_data_->device;
    
    if (optical_flow_pipeline_.pipeline) {
        dt.DestroyPipeline(device, optical_flow_pipeline_.pipeline, nullptr);
        dt.DestroyPipelineLayout(device, optical_flow_pipeline_.layout, nullptr);
        dt.DestroyDescriptorSetLayout(device, optical_flow_pipeline_.desc_set_layout, nullptr);
        dt.DestroyDescriptorPool(device, optical_flow_pipeline_.desc_pool, nullptr);
    }
    
    if (interpolation_pipeline_.pipeline) {
        dt.DestroyPipeline(device, interpolation_pipeline_.pipeline, nullptr);
        dt.DestroyPipelineLayout(device, interpolation_pipeline_.layout, nullptr);
        dt.DestroyDescriptorSetLayout(device, interpolation_pipeline_.desc_set_layout, nullptr);
        dt.DestroyDescriptorPool(device, interpolation_pipeline_.desc_pool, nullptr);
    }
    
    if (motion_estimation_pipeline_.pipeline) {
        dt.DestroyPipeline(device, motion_estimation_pipeline_.pipeline, nullptr);
        dt.DestroyPipelineLayout(device, motion_estimation_pipeline_.layout, nullptr);
        dt.DestroyDescriptorSetLayout(device, motion_estimation_pipeline_.desc_set_layout, nullptr);
        dt.DestroyDescriptorPool(device, motion_estimation_pipeline_.desc_pool, nullptr);
    }
    
    // Destroy shader modules
    if (optical_flow_shader_) dt.DestroyShaderModule(device, optical_flow_shader_, nullptr);
    if (interpolation_shader_) dt.DestroyShaderModule(device, interpolation_shader_, nullptr);
    if (motion_estimation_shader_) dt.DestroyShaderModule(device, motion_estimation_shader_, nullptr);
    
    // Destroy constants buffer
    if (constants_buffer_) {
        dt.DestroyBuffer(device, constants_buffer_, nullptr);
        dt.FreeMemory(device, constants_memory_, nullptr);
    }
}

void FrameInterpolator::initSwapchain(VkSwapchainKHR swapchain, VkFormat format,
                                    VkExtent2D extent, uint32_t image_count) {
    std::lock_guard<std::mutex> lock(resources_mutex_);
    
    // Create swapchain resources
    auto resources = std::make_unique<SwapchainResources>();
    resources->format = format;
    resources->extent = extent;
    resources->image_count = image_count;
    
    // Allocate intermediate buffers (3 for triple buffering)
    const uint32_t buffer_count = 3;
    resources->motion_vector_buffers.resize(buffer_count);
    resources->optical_flow_buffers.resize(buffer_count);
    resources->interpolated_buffers.resize(buffer_count);
    
    // Create buffers
    for (uint32_t i = 0; i < buffer_count; i++) {
        // Motion vectors (R16G16_SFLOAT)
        createFrameBuffer(resources->motion_vector_buffers[i],
                        VK_FORMAT_R16G16_SFLOAT, extent,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        
        // Optical flow (R16G16_SFLOAT)
        createFrameBuffer(resources->optical_flow_buffers[i],
                        VK_FORMAT_R16G16_SFLOAT, extent,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        
        // Interpolated frames (same format as swapchain)
        createFrameBuffer(resources->interpolated_buffers[i],
                        format, extent,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    }
    
    // Allocate command buffers
    VkCommandBufferAllocateInfo cmd_alloc_info = {};
    cmd_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc_info.commandPool = device_data_->compute_command_pool;
    cmd_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc_info.commandBufferCount = buffer_count;
    
    resources->command_buffers.resize(buffer_count);
    device_data_->dispatch_table.AllocateCommandBuffers(
        device_data_->device, &cmd_alloc_info, resources->command_buffers.data());
    
    // Create descriptor sets
    createDescriptorSets(*resources);
    
    swapchain_resources_[swapchain] = std::move(resources);
    
    if (device_data_->instance_data->debug_enabled) {
        std::cout << "[Frame Interpolation] Initialized swapchain resources\n";
        std::cout << "  Buffer count: " << buffer_count << "\n";
    }
}

void FrameInterpolator::cleanupSwapchain(VkSwapchainKHR swapchain) {
    std::lock_guard<std::mutex> lock(resources_mutex_);
    
    auto it = swapchain_resources_.find(swapchain);
    if (it == swapchain_resources_.end()) return;
    
    auto& resources = it->second;
    auto& dt = device_data_->dispatch_table;
    VkDevice device = device_data_->device;
    
    // Free command buffers
    if (!resources->command_buffers.empty()) {
        dt.FreeCommandBuffers(device, device_data_->compute_command_pool,
                            resources->command_buffers.size(),
                            resources->command_buffers.data());
    }
    
    // Destroy frame buffers
    for (auto& buffer : resources->motion_vector_buffers) {
        destroyFrameBuffer(buffer);
    }
    for (auto& buffer : resources->optical_flow_buffers) {
        destroyFrameBuffer(buffer);
    }
    for (auto& buffer : resources->interpolated_buffers) {
        destroyFrameBuffer(buffer);
    }
    
    swapchain_resources_.erase(it);
}

VkImage FrameInterpolator::interpolateFrame(const InterpolationRequest& request) {
    // Queue the request for async processing
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        request_queue_.push(request);
    }
    queue_cv_.notify_one();
    
    // For now, return null - in a real implementation, this would
    // return a future or handle to track the async operation
    return VK_NULL_HANDLE;
}

void FrameInterpolator::createConstantsBuffer() {
    VkDevice device = device_data_->device;
    auto& dt = device_data_->dispatch_table;
    
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = sizeof(InterpolationConstants);
    buffer_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    dt.CreateBuffer(device, &buffer_info, nullptr, &constants_buffer_);
    
    // Allocate memory
    VkMemoryRequirements mem_reqs;
    dt.GetBufferMemoryRequirements(device, constants_buffer_, &mem_reqs);
    
    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = layer_utils::findMemoryType(
        device_data_->memory_properties,
        mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    dt.AllocateMemory(device, &alloc_info, nullptr, &constants_memory_);
    dt.BindBufferMemory(device, constants_buffer_, constants_memory_, 0);
    
    // Initialize constants
    InterpolationConstants* constants;
    dt.MapMemory(device, constants_memory_, 0, sizeof(InterpolationConstants),
                0, (void**)&constants);
    
    constants->motion_scale[0] = 1.0f;
    constants->motion_scale[1] = 1.0f;
    constants->interpolation_factor = 0.5f;
    constants->depth_threshold = 0.01f;
    constants->motion_threshold = 0.5f;
    constants->disocclusion_threshold = 0.1f;
    constants->block_size = 8;
    constants->search_radius = 16;
    constants->quality_level = LayerData::getInstance().settings.quality;
    
    dt.UnmapMemory(device, constants_memory_);
}

void FrameInterpolator::createShaderModules() {
    // In a real implementation, these would load compiled SPIR-V shaders
    // For now, we'll use placeholder paths
    optical_flow_shader_ = loadShaderModule("shaders/optical_flow.spv");
    interpolation_shader_ = loadShaderModule("shaders/interpolate_frame.spv");
    motion_estimation_shader_ = loadShaderModule("shaders/motion_estimation.spv");
}

VkShaderModule FrameInterpolator::loadShaderModule(const std::string& path) {
    VkDevice device = device_data_->device;
    auto& dt = device_data_->dispatch_table;
    
    try {
        auto code = loadSPIRV(path);
        
        VkShaderModuleCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        create_info.codeSize = code.size() * sizeof(uint32_t);
        create_info.pCode = code.data();
        
        VkShaderModule module;
        if (dt.CreateShaderModule(device, &create_info, nullptr, &module) != VK_SUCCESS) {
            std::cerr << "[Frame Interpolation] Warning: Failed to create shader module from " 
                      << path << "\n";
            return VK_NULL_HANDLE;
        }
        
        return module;
    } catch (const std::exception& e) {
        std::cerr << "[Frame Interpolation] Warning: " << e.what() << "\n";
        return VK_NULL_HANDLE;
    }
}

void FrameInterpolator::createPipelines() {
    VkDevice device = device_data_->device;
    auto& dt = device_data_->dispatch_table;
    
    // Create descriptor set layouts
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    
    // Optical flow pipeline layout
    bindings = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // prev frame
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // curr frame
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // output flow
        {3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // constants
    };
    
    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = bindings.size();
    layout_info.pBindings = bindings.data();
    
    dt.CreateDescriptorSetLayout(device, &layout_info, nullptr, 
                               &optical_flow_pipeline_.desc_set_layout);
    
    // Create pipeline layouts
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &optical_flow_pipeline_.desc_set_layout;
    
    dt.CreatePipelineLayout(device, &pipeline_layout_info, nullptr,
                          &optical_flow_pipeline_.layout);
    
    // Create compute pipelines
    if (optical_flow_shader_) {
        VkPipelineShaderStageCreateInfo stage_info = {};
        stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage_info.module = optical_flow_shader_;
        stage_info.pName = "main";
        
        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = stage_info;
        pipeline_info.layout = optical_flow_pipeline_.layout;
        
        dt.CreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                nullptr, &optical_flow_pipeline_.pipeline);
    }
    
    // Similar setup for interpolation and motion estimation pipelines...
    // (Simplified for brevity)
    
    // Create descriptor pool
    std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 100},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 100},
    };
    
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = pool_sizes.size();
    pool_info.pPoolSizes = pool_sizes.data();
    pool_info.maxSets = 100;
    
    dt.CreateDescriptorPool(device, &pool_info, nullptr,
                          &optical_flow_pipeline_.desc_pool);
}

void FrameInterpolator::createFrameBuffer(SwapchainResources::FrameBuffer& buffer,
                                        VkFormat format, VkExtent2D extent,
                                        VkImageUsageFlags usage) {
    VkDevice device = device_data_->device;
    auto& dt = device_data_->dispatch_table;
    
    // Create image
    VkImageCreateInfo image_info = {};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent.width = extent.width;
    image_info.extent.height = extent.height;
    image_info.extent.depth = 1;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = format;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.usage = usage;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    dt.CreateImage(device, &image_info, nullptr, &buffer.image);
    
    // Allocate memory
    VkMemoryRequirements mem_reqs;
    dt.GetImageMemoryRequirements(device, buffer.image, &mem_reqs);
    
    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = layer_utils::findMemoryType(
        device_data_->memory_properties,
        mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    dt.AllocateMemory(device, &alloc_info, nullptr, &buffer.memory);
    dt.BindImageMemory(device, buffer.image, buffer.memory, 0);
    
    // Create image view
    buffer.view = layer_utils::createImageView(device, buffer.image, format,
                                              VK_IMAGE_ASPECT_COLOR_BIT);
}

void FrameInterpolator::destroyFrameBuffer(SwapchainResources::FrameBuffer& buffer) {
    VkDevice device = device_data_->device;
    auto& dt = device_data_->dispatch_table;
    
    if (buffer.view) dt.DestroyImageView(device, buffer.view, nullptr);
    if (buffer.image) dt.DestroyImage(device, buffer.image, nullptr);
    if (buffer.memory) dt.FreeMemory(device, buffer.memory, nullptr);
    
    buffer.view = VK_NULL_HANDLE;
    buffer.image = VK_NULL_HANDLE;
    buffer.memory = VK_NULL_HANDLE;
}

void FrameInterpolator::createDescriptorSets(SwapchainResources& resources) {
    VkDevice device = device_data_->device;
    auto& dt = device_data_->dispatch_table;
    
    const uint32_t buffer_count = resources.motion_vector_buffers.size();
    
    // Allocate descriptor sets
    resources.optical_flow_desc_sets.resize(buffer_count);
    resources.interpolation_desc_sets.resize(buffer_count);
    resources.motion_estimation_desc_sets.resize(buffer_count);
    
    std::vector<VkDescriptorSetLayout> layouts(buffer_count, 
                                              optical_flow_pipeline_.desc_set_layout);
    
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = optical_flow_pipeline_.desc_pool;
    alloc_info.descriptorSetCount = buffer_count;
    alloc_info.pSetLayouts = layouts.data();
    
    if (optical_flow_pipeline_.desc_pool) {
        dt.AllocateDescriptorSets(device, &alloc_info, 
                                resources.optical_flow_desc_sets.data());
    }
    
    // Similar allocation for other descriptor sets...
}

void FrameInterpolator::updateDescriptorSets(SwapchainResources& resources, 
                                           uint32_t buffer_index,
                                           const InterpolationRequest& request) {
    VkDevice device = device_data_->device;
    auto& dt = device_data_->dispatch_table;
    
    // Create sampler if needed
    static VkSampler linear_sampler = VK_NULL_HANDLE;
    if (!linear_sampler) {
        VkSamplerCreateInfo sampler_info = {};
        sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        dt.CreateSampler(device, &sampler_info, nullptr, &linear_sampler);
    }
    
    // Update optical flow descriptor set
    std::vector<VkWriteDescriptorSet> writes;
    
    VkDescriptorImageInfo prev_frame_info = {};
    prev_frame_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    prev_frame_info.imageView = request.prev_frame_view;
    prev_frame_info.sampler = linear_sampler;
    
    VkDescriptorImageInfo curr_frame_info = {};
    curr_frame_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    curr_frame_info.imageView = request.curr_frame_view;
    curr_frame_info.sampler = linear_sampler;
    
    VkDescriptorImageInfo output_flow_info = {};
    output_flow_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    output_flow_info.imageView = resources.optical_flow_buffers[buffer_index].view;
    
    VkDescriptorBufferInfo constants_info = {};
    constants_info.buffer = constants_buffer_;
    constants_info.offset = 0;
    constants_info.range = sizeof(InterpolationConstants);
    
    if (!resources.optical_flow_desc_sets.empty()) {
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET});
        writes.back().dstSet = resources.optical_flow_desc_sets[buffer_index];
        writes.back().dstBinding = 0;
        writes.back().descriptorCount = 1;
        writes.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes.back().pImageInfo = &prev_frame_info;
        
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET});
        writes.back().dstSet = resources.optical_flow_desc_sets[buffer_index];
        writes.back().dstBinding = 1;
        writes.back().descriptorCount = 1;
        writes.back().descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes.back().pImageInfo = &curr_frame_info;
        
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET});
        writes.back().dstSet = resources.optical_flow_desc_sets[buffer_index];
        writes.back().dstBinding = 2;
        writes.back().descriptorCount = 1;
        writes.back().descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes.back().pImageInfo = &output_flow_info;
        
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET});
        writes.back().dstSet = resources.optical_flow_desc_sets[buffer_index];
        writes.back().dstBinding = 3;
        writes.back().descriptorCount = 1;
        writes.back().descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes.back().pBufferInfo = &constants_info;
        
        dt.UpdateDescriptorSets(device, writes.size(), writes.data(), 0, nullptr);
    }
}

void FrameInterpolator::computeThreadFunc() {
    while (true) {
        InterpolationRequest request;
        
        // Wait for request
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !request_queue_.empty() || should_stop_; });
            
            if (should_stop_) break;
            
            request = request_queue_.front();
            request_queue_.pop();
        }
        
        // Process the request
        processInterpolationRequest(request);
    }
}

void FrameInterpolator::processInterpolationRequest(const InterpolationRequest& request) {
    std::lock_guard<std::mutex> lock(resources_mutex_);
    
    auto it = swapchain_resources_.find(request.swapchain);
    if (it == swapchain_resources_.end()) return;
    
    auto& resources = *it->second;
    auto& dt = device_data_->dispatch_table;
    VkDevice device = device_data_->device;
    
    // Get current buffer index
    uint32_t buffer_index = resources.current_buffer_index;
    resources.current_buffer_index = (buffer_index + 1) % resources.command_buffers.size();
    
    // Update descriptor sets
    updateDescriptorSets(resources, buffer_index, request);
    
    // Update constants
    InterpolationConstants* constants;
    dt.MapMemory(device, constants_memory_, 0, sizeof(InterpolationConstants),
                0, (void**)&constants);
    constants->frame_width = resources.extent.width;
    constants->frame_height = resources.extent.height;
    constants->interpolation_factor = request.interpolation_factor;
    dt.UnmapMemory(device, constants_memory_);
    
    // Record command buffer
    VkCommandBuffer cmd = resources.command_buffers[buffer_index];
    
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    dt.BeginCommandBuffer(cmd, &begin_info);
    
    // Transition images to correct layouts
    VkImageSubresourceRange color_range = {};
    color_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    color_range.levelCount = 1;
    color_range.layerCount = 1;
    
    layer_utils::transitionImageLayout(cmd, request.prev_frame,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        color_range, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    
    layer_utils::transitionImageLayout(cmd, request.curr_frame,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        color_range, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    
    // Compute optical flow
    if (optical_flow_pipeline_.pipeline) {
        computeOpticalFlow(cmd, request.prev_frame_view, request.curr_frame_view,
                         resources.optical_flow_buffers[buffer_index].view,
                         resources.extent);
    }
    
    // Interpolate frame
    if (interpolation_pipeline_.pipeline) {
        interpolateFrameCompute(cmd,
            request.prev_frame_view, request.curr_frame_view,
            resources.optical_flow_buffers[buffer_index].view,
            resources.motion_vector_buffers[buffer_index].view,
            resources.interpolated_buffers[buffer_index].view,
            request.interpolation_factor,
            resources.extent);
    }
    
    // Transition images back
    layer_utils::transitionImageLayout(cmd, request.prev_frame,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        color_range, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    
    layer_utils::transitionImageLayout(cmd, request.curr_frame,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        color_range, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    
    dt.EndCommandBuffer(cmd);
    
    // Submit to compute queue
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;
    
    dt.QueueSubmit(device_data_->compute_queue, 1, &submit_info, VK_NULL_HANDLE);
    
    // In a real implementation, we would signal completion and make the
    // interpolated frame available for presentation
}

void FrameInterpolator::initializeFidelityFX() {
    // Initialize AMD FidelityFX context
    // This would set up the FidelityFX SDK for optical flow and frame interpolation
    if (device_data_->instance_data->debug_enabled) {
        std::cout << "[Frame Interpolation] Initializing FidelityFX\n";
    }
}

void FrameInterpolator::shutdownFidelityFX() {
    // Cleanup FidelityFX resources
}

void FrameInterpolator::computeOpticalFlow(VkCommandBuffer cmd,
                                         VkImageView prev_frame, VkImageView curr_frame,
                                         VkImageView output_flow,
                                         VkExtent2D extent) {
    auto& dt = device_data_->dispatch_table;
    
    if (!optical_flow_pipeline_.pipeline) return;
    
    // Bind pipeline and descriptor sets
    dt.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, 
                      optical_flow_pipeline_.pipeline);
    
    // The actual descriptor set would be properly set up with the views
    // dt.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
    //                         optical_flow_pipeline_.layout, 0, 1,
    //                         &desc_set, 0, nullptr);
    
    // Dispatch compute shader
    uint32_t group_count_x = (extent.width + 7) / 8;
    uint32_t group_count_y = (extent.height + 7) / 8;
    dt.CmdDispatch(cmd, group_count_x, group_count_y, 1);
}

void FrameInterpolator::interpolateFrameCompute(VkCommandBuffer cmd,
                                              VkImageView prev_frame, VkImageView curr_frame,
                                              VkImageView optical_flow, VkImageView motion_vectors,
                                              VkImageView output_frame,
                                              float interpolation_factor,
                                              VkExtent2D extent) {
    auto& dt = device_data_->dispatch_table;
    
    if (!interpolation_pipeline_.pipeline) return;
    
    // Similar to optical flow, bind pipeline and dispatch
    dt.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      interpolation_pipeline_.pipeline);
    
    uint32_t group_count_x = (extent.width + 7) / 8;
    uint32_t group_count_y = (extent.height + 7) / 8;
    dt.CmdDispatch(cmd, group_count_x, group_count_y, 1);
}
