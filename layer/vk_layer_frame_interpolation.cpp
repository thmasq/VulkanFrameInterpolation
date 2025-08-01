#include "vk_layer_frame_interpolation.h"
#include "frame_interpolator.h"
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <chrono>

// Layer dispatch table setup
#define DISPATCH_TABLE_ENTRY(table, instance, name) \
    table.name = (PFN_vk##name)vkGetInstanceProcAddr(instance, "vk" #name)

#define DEVICE_DISPATCH_TABLE_ENTRY(table, device, name) \
    table.name = (PFN_vk##name)vkGetDeviceProcAddr(device, "vk" #name)

// Global layer data implementation
LayerData::LayerData() {
    loadSettings();
}

void LayerData::loadSettings() {
    // Load settings from environment variables
    const char* enabled = std::getenv("VK_FRAME_INTERPOLATION_ENABLED");
    if (enabled && strcmp(enabled, "0") == 0) {
        settings.enabled = false;
    }
    
    const char* fps = std::getenv("VK_FRAME_INTERPOLATION_TARGET_FPS");
    if (fps) {
        settings.target_fps = std::atoi(fps);
    }
    
    const char* quality = std::getenv("VK_FRAME_INTERPOLATION_QUALITY");
    if (quality) {
        int q = std::atoi(quality);
        if (q >= 0 && q <= 2) {
            settings.quality = q;
        }
    }
    
    const char* debug = std::getenv("VK_FRAME_INTERPOLATION_DEBUG");
    if (debug && strcmp(debug, "1") == 0) {
        settings.debug = true;
    }
    
    const char* stats = std::getenv("VK_FRAME_INTERPOLATION_SHOW_STATS");
    if (stats && strcmp(stats, "1") == 0) {
        settings.show_stats = true;
    }
}

void LayerData::addInstance(VkInstance instance, InstanceData* data) {
    std::lock_guard<std::mutex> lock(instance_mutex);
    instances[instance] = std::unique_ptr<InstanceData>(data);
}

void LayerData::removeInstance(VkInstance instance) {
    std::lock_guard<std::mutex> lock(instance_mutex);
    instances.erase(instance);
}

InstanceData* LayerData::getInstance(VkInstance instance) {
    std::lock_guard<std::mutex> lock(instance_mutex);
    auto it = instances.find(instance);
    return (it != instances.end()) ? it->second.get() : nullptr;
}

void LayerData::addDevice(VkDevice device, DeviceData* data) {
    std::lock_guard<std::mutex> lock(device_mutex);
    devices[device] = std::unique_ptr<DeviceData>(data);
}

void LayerData::removeDevice(VkDevice device) {
    std::lock_guard<std::mutex> lock(device_mutex);
    devices.erase(device);
}

DeviceData* LayerData::getDevice(VkDevice device) {
    std::lock_guard<std::mutex> lock(device_mutex);
    auto it = devices.find(device);
    return (it != devices.end()) ? it->second.get() : nullptr;
}

void LayerData::addSwapchain(VkSwapchainKHR swapchain, SwapchainData* data) {
    std::lock_guard<std::mutex> lock(swapchain_mutex);
    swapchains[swapchain] = std::unique_ptr<SwapchainData>(data);
}

void LayerData::removeSwapchain(VkSwapchainKHR swapchain) {
    std::lock_guard<std::mutex> lock(swapchain_mutex);
    swapchains.erase(swapchain);
}

SwapchainData* LayerData::getSwapchain(VkSwapchainKHR swapchain) {
    std::lock_guard<std::mutex> lock(swapchain_mutex);
    auto it = swapchains.find(swapchain);
    return (it != swapchains.end()) ? it->second.get() : nullptr;
}

// Intercepted functions
extern "C" {

// vkCreateInstance
VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(
    const VkInstanceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkInstance* pInstance)
{
    VkLayerInstanceCreateInfo* chain_info = 
        (VkLayerInstanceCreateInfo*)pCreateInfo->pNext;
    
    while (chain_info && 
           chain_info->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO) {
        chain_info = (VkLayerInstanceCreateInfo*)chain_info->pNext;
    }
    
    if (!chain_info || !chain_info->u.pLayerInfo) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = 
        chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkCreateInstance fpCreateInstance = 
        (PFN_vkCreateInstance)fpGetInstanceProcAddr(nullptr, "vkCreateInstance");
    
    if (!fpCreateInstance) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    // Advance the chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;
    
    VkResult result = fpCreateInstance(pCreateInfo, pAllocator, pInstance);
    if (result != VK_SUCCESS) {
        return result;
    }
    
    // Create instance data
    auto instance_data = new InstanceData();
    instance_data->instance = *pInstance;
    instance_data->debug_enabled = LayerData::getInstance().settings.debug;
    
    // Initialize dispatch table
    VkLayerInstanceDispatchTable& table = instance_data->dispatch_table;
    DISPATCH_TABLE_ENTRY(table, *pInstance, DestroyInstance);
    DISPATCH_TABLE_ENTRY(table, *pInstance, GetPhysicalDeviceProperties);
    DISPATCH_TABLE_ENTRY(table, *pInstance, GetPhysicalDeviceMemoryProperties);
    DISPATCH_TABLE_ENTRY(table, *pInstance, GetPhysicalDeviceQueueFamilyProperties);
    DISPATCH_TABLE_ENTRY(table, *pInstance, CreateDevice);
    DISPATCH_TABLE_ENTRY(table, *pInstance, EnumerateDeviceExtensionProperties);
    
    LayerData::getInstance().addInstance(*pInstance, instance_data);
    
    if (instance_data->debug_enabled) {
        std::cout << "[Frame Interpolation] Instance created\n";
    }
    
    return VK_SUCCESS;
}

// vkDestroyInstance
VKAPI_ATTR void VKAPI_CALL vkDestroyInstance(
    VkInstance instance,
    const VkAllocationCallbacks* pAllocator)
{
    auto instance_data = LayerData::getInstance().getInstance(instance);
    if (!instance_data) return;
    
    instance_data->dispatch_table.DestroyInstance(instance, pAllocator);
    LayerData::getInstance().removeInstance(instance);
}

// vkCreateDevice
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(
    VkPhysicalDevice physicalDevice,
    const VkDeviceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDevice* pDevice)
{
    VkLayerDeviceCreateInfo* chain_info = 
        (VkLayerDeviceCreateInfo*)pCreateInfo->pNext;
    
    while (chain_info && 
           chain_info->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO) {
        chain_info = (VkLayerDeviceCreateInfo*)chain_info->pNext;
    }
    
    if (!chain_info || !chain_info->u.pLayerInfo) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = 
        chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr fpGetDeviceProcAddr = 
        chain_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    PFN_vkCreateDevice fpCreateDevice = 
        (PFN_vkCreateDevice)fpGetInstanceProcAddr(nullptr, "vkCreateDevice");
    
    if (!fpCreateDevice) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    // Find instance data
    VkInstance instance = VK_NULL_HANDLE;
    auto& layer_data = LayerData::getInstance();
    
    // Get instance from physical device (this is a bit hacky but works)
    for (auto& [inst, inst_data] : layer_data.instances) {
        VkInstance temp_instance = inst;
        // We'll assume the physical device belongs to the most recent instance
        // In practice, this works fine for single-instance applications
        instance = temp_instance;
    }
    
    auto instance_data = layer_data.getInstance(instance);
    if (!instance_data) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    // Advance the chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;
    
    // Modify device creation info to request additional queues/features if needed
    VkDeviceCreateInfo modified_create_info = *pCreateInfo;
    
    VkResult result = fpCreateDevice(physicalDevice, &modified_create_info, 
                                   pAllocator, pDevice);
    if (result != VK_SUCCESS) {
        return result;
    }
    
    // Create device data
    auto device_data = new DeviceData();
    device_data->device = *pDevice;
    device_data->physical_device = physicalDevice;
    device_data->instance_data = instance_data;
    
    // Get device properties
    instance_data->dispatch_table.GetPhysicalDeviceProperties(
        physicalDevice, &device_data->properties);
    instance_data->dispatch_table.GetPhysicalDeviceMemoryProperties(
        physicalDevice, &device_data->memory_properties);
    
    // Find queue families
    uint32_t queue_family_count = 0;
    instance_data->dispatch_table.GetPhysicalDeviceQueueFamilyProperties(
        physicalDevice, &queue_family_count, nullptr);
    
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    instance_data->dispatch_table.GetPhysicalDeviceQueueFamilyProperties(
        physicalDevice, &queue_family_count, queue_families.data());
    
    device_data->graphics_queue_family = UINT32_MAX;
    device_data->compute_queue_family = UINT32_MAX;
    device_data->present_queue_family = UINT32_MAX;
    
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            device_data->graphics_queue_family = i;
            device_data->present_queue_family = i; // Assume graphics can present
        }
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            device_data->compute_queue_family = i;
        }
    }
    
    // Initialize device dispatch table
    VkLayerDispatchTable& table = device_data->dispatch_table;
    layer_init_device_dispatch_table(*pDevice, &table, fpGetDeviceProcAddr);
    
    // Get compute queue
    if (device_data->compute_queue_family != UINT32_MAX) {
        table.GetDeviceQueue(*pDevice, device_data->compute_queue_family, 0, 
                           &device_data->compute_queue);
        
        // Create command pool for compute commands
        VkCommandPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = device_data->compute_queue_family;
        
        table.CreateCommandPool(*pDevice, &pool_info, nullptr, 
                              &device_data->compute_command_pool);
    }
    
    // Create frame interpolator
    device_data->interpolator = std::make_unique<FrameInterpolator>(device_data);
    
    layer_data.addDevice(*pDevice, device_data);
    
    if (instance_data->debug_enabled) {
        std::cout << "[Frame Interpolation] Device created\n";
        std::cout << "  Graphics queue family: " << device_data->graphics_queue_family << "\n";
        std::cout << "  Compute queue family: " << device_data->compute_queue_family << "\n";
    }
    
    return VK_SUCCESS;
}

// vkDestroyDevice
VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(
    VkDevice device,
    const VkAllocationCallbacks* pAllocator)
{
    auto device_data = LayerData::getInstance().getDevice(device);
    if (!device_data) return;
    
    // Wait for device idle before cleanup
    device_data->dispatch_table.DeviceWaitIdle(device);
    
    // Destroy command pool
    if (device_data->compute_command_pool != VK_NULL_HANDLE) {
        device_data->dispatch_table.DestroyCommandPool(
            device, device_data->compute_command_pool, nullptr);
    }
    
    // Destroy interpolator
    device_data->interpolator.reset();
    
    device_data->dispatch_table.DestroyDevice(device, pAllocator);
    LayerData::getInstance().removeDevice(device);
}

// vkCreateSwapchainKHR
VKAPI_ATTR VkResult VKAPI_CALL vkCreateSwapchainKHR(
    VkDevice device,
    const VkSwapchainCreateInfoKHR* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkSwapchainKHR* pSwapchain)
{
    auto device_data = LayerData::getInstance().getDevice(device);
    if (!device_data) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    VkResult result = device_data->dispatch_table.CreateSwapchainKHR(
        device, pCreateInfo, pAllocator, pSwapchain);
    if (result != VK_SUCCESS) {
        return result;
    }
    
    // Create swapchain data
    auto swapchain_data = new SwapchainData();
    swapchain_data->swapchain = *pSwapchain;
    swapchain_data->device_data = device_data;
    swapchain_data->format = pCreateInfo->imageFormat;
    swapchain_data->extent = pCreateInfo->imageExtent;
    swapchain_data->history_write_index = 0;
    
    // Get swapchain images
    device_data->dispatch_table.GetSwapchainImagesKHR(
        device, *pSwapchain, &swapchain_data->image_count, nullptr);
    swapchain_data->images.resize(swapchain_data->image_count);
    device_data->dispatch_table.GetSwapchainImagesKHR(
        device, *pSwapchain, &swapchain_data->image_count, 
        swapchain_data->images.data());
    
    // Create image views
    swapchain_data->image_views.reserve(swapchain_data->image_count);
    for (auto image : swapchain_data->images) {
        VkImageView view = layer_utils::createImageView(
            device, image, swapchain_data->format, VK_IMAGE_ASPECT_COLOR_BIT);
        swapchain_data->image_views.push_back(view);
    }
    
    // Create synchronization objects
    swapchain_data->interpolation_complete_semaphores.reserve(swapchain_data->image_count);
    swapchain_data->interpolation_fences.reserve(swapchain_data->image_count);
    
    for (uint32_t i = 0; i < swapchain_data->image_count; i++) {
        swapchain_data->interpolation_complete_semaphores.push_back(
            layer_utils::createSemaphore(device));
        swapchain_data->interpolation_fences.push_back(
            layer_utils::createFence(device, true));
    }
    
    // Initialize frame history
    for (auto& frame : swapchain_data->frame_history) {
        frame.valid = false;
        frame.ready_semaphore = layer_utils::createSemaphore(device);
    }
    
    // Initialize interpolator for this swapchain
    if (device_data->interpolator) {
        device_data->interpolator->initSwapchain(
            *pSwapchain, swapchain_data->format, 
            swapchain_data->extent, swapchain_data->image_count);
    }
    
    LayerData::getInstance().addSwapchain(*pSwapchain, swapchain_data);
    
    if (device_data->instance_data->debug_enabled) {
        std::cout << "[Frame Interpolation] Swapchain created\n";
        std::cout << "  Format: " << swapchain_data->format << "\n";
        std::cout << "  Extent: " << swapchain_data->extent.width 
                  << "x" << swapchain_data->extent.height << "\n";
        std::cout << "  Image count: " << swapchain_data->image_count << "\n";
    }
    
    return VK_SUCCESS;
}

// vkDestroySwapchainKHR
VKAPI_ATTR void VKAPI_CALL vkDestroySwapchainKHR(
    VkDevice device,
    VkSwapchainKHR swapchain,
    const VkAllocationCallbacks* pAllocator)
{
    auto device_data = LayerData::getInstance().getDevice(device);
    auto swapchain_data = LayerData::getInstance().getSwapchain(swapchain);
    
    if (!device_data || !swapchain_data) return;
    
    // Wait for any pending interpolation
    device_data->dispatch_table.DeviceWaitIdle(device);
    
    // Cleanup interpolator resources for this swapchain
    if (device_data->interpolator) {
        device_data->interpolator->cleanupSwapchain(swapchain);
    }
    
    // Destroy image views
    for (auto view : swapchain_data->image_views) {
        device_data->dispatch_table.DestroyImageView(device, view, nullptr);
    }
    
    // Destroy synchronization objects
    for (auto semaphore : swapchain_data->interpolation_complete_semaphores) {
        device_data->dispatch_table.DestroySemaphore(device, semaphore, nullptr);
    }
    for (auto fence : swapchain_data->interpolation_fences) {
        device_data->dispatch_table.DestroyFence(device, fence, nullptr);
    }
    for (auto& frame : swapchain_data->frame_history) {
        if (frame.ready_semaphore != VK_NULL_HANDLE) {
            device_data->dispatch_table.DestroySemaphore(
                device, frame.ready_semaphore, nullptr);
        }
    }
    
    device_data->dispatch_table.DestroySwapchainKHR(device, swapchain, pAllocator);
    LayerData::getInstance().removeSwapchain(swapchain);
}

// vkQueuePresentKHR - The main interception point
VKAPI_ATTR VkResult VKAPI_CALL vkQueuePresentKHR(
    VkQueue queue,
    const VkPresentInfoKHR* pPresentInfo)
{
    auto& layer_data = LayerData::getInstance();
    
    // Find device data from queue
    DeviceData* device_data = nullptr;
    for (auto& [device, data] : layer_data.devices) {
        // This is a simplified approach - in production, you'd want a better way
        // to map queues to devices
        device_data = data.get();
        break;
    }
    
    if (!device_data || !layer_data.settings.enabled) {
        // Pass through if layer is disabled or data not found
        return device_data ? device_data->dispatch_table.QueuePresentKHR(queue, pPresentInfo) 
                          : VK_ERROR_DEVICE_LOST;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Process each swapchain in the present info
    for (uint32_t i = 0; i < pPresentInfo->swapchainCount; i++) {
        VkSwapchainKHR swapchain = pPresentInfo->pSwapchains[i];
        uint32_t image_index = pPresentInfo->pImageIndices[i];
        
        auto swapchain_data = layer_data.getSwapchain(swapchain);
        if (!swapchain_data) continue;
        
        // Increment frame number
        uint64_t current_frame = swapchain_data->frame_number.fetch_add(1);
        
        // Update frame history
        auto& history = swapchain_data->frame_history;
        uint32_t hist_idx = swapchain_data->history_write_index;
        
        history[hist_idx].image = swapchain_data->images[image_index];
        history[hist_idx].view = swapchain_data->image_views[image_index];
        history[hist_idx].frame_number = current_frame;
        history[hist_idx].valid = true;
        
        swapchain_data->history_write_index = 
            (swapchain_data->history_write_index + 1) % SwapchainData::FRAME_HISTORY_SIZE;
        
        // Check if we have enough frames for interpolation
        int valid_frames = 0;
        for (const auto& frame : history) {
            if (frame.valid) valid_frames++;
        }
        
        if (valid_frames >= 2 && device_data->interpolator) {
            // Find the two most recent frames
            uint32_t prev_idx = (hist_idx + SwapchainData::FRAME_HISTORY_SIZE - 1) 
                              % SwapchainData::FRAME_HISTORY_SIZE;
            
            // Only interpolate every other frame to achieve 2x framerate
            if (current_frame % 2 == 1) {
                // Create interpolation request
                FrameInterpolator::InterpolationRequest request;
                request.swapchain = swapchain;
                request.prev_frame = history[prev_idx].image;
                request.prev_frame_view = history[prev_idx].view;
                request.curr_frame = history[hist_idx].image;
                request.curr_frame_view = history[hist_idx].view;
                request.output_image = VK_NULL_HANDLE; // Will be allocated by interpolator
                request.frame_number = current_frame;
                request.interpolation_factor = 0.5f; // Halfway between frames
                
                // Submit interpolation work
                VkImage interpolated_image = device_data->interpolator->interpolateFrame(request);
                
                if (interpolated_image != VK_NULL_HANDLE) {
                    swapchain_data->stats.frames_interpolated.fetch_add(1);
                    
                    // TODO: Present the interpolated frame
                    // This requires modifying the present queue to alternate between
                    // real and interpolated frames
                }
            }
        }
        
        swapchain_data->stats.frames_presented.fetch_add(1);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    // Update stats
    if (pPresentInfo->swapchainCount > 0) {
        auto swapchain_data = layer_data.getSwapchain(pPresentInfo->pSwapchains[0]);
        if (swapchain_data) {
            swapchain_data->stats.interpolation_time_us.store(duration);
        }
    }
    
    // Call the driver's present function
    return device_data->dispatch_table.QueuePresentKHR(queue, pPresentInfo);
}

// Layer discovery functions
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(
    uint32_t* pPropertyCount,
    VkLayerProperties* pProperties)
{
    if (pPropertyCount) *pPropertyCount = 1;
    
    if (pProperties) {
        strcpy(pProperties->layerName, LAYER_NAME);
        strcpy(pProperties->description, LAYER_DESCRIPTION);
        pProperties->implementationVersion = 1;
        pProperties->specVersion = VK_API_VERSION_1_0;
    }
    
    return VK_SUCCESS;
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(
    VkPhysicalDevice physicalDevice,
    uint32_t* pPropertyCount,
    VkLayerProperties* pProperties)
{
    return vkEnumerateInstanceLayerProperties(pPropertyCount, pProperties);
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(
    const char* pLayerName,
    uint32_t* pPropertyCount,
    VkExtensionProperties* pProperties)
{
    if (pLayerName && strcmp(pLayerName, LAYER_NAME) == 0) {
        // This layer doesn't expose any instance extensions
        if (pPropertyCount) *pPropertyCount = 0;
        return VK_SUCCESS;
    }
    
    return VK_ERROR_LAYER_NOT_PRESENT;
}

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice physicalDevice,
    const char* pLayerName,
    uint32_t* pPropertyCount,
    VkExtensionProperties* pProperties)
{
    if (pLayerName && strcmp(pLayerName, LAYER_NAME) == 0) {
        // This layer doesn't expose any device extensions
        if (pPropertyCount) *pPropertyCount = 0;
        return VK_SUCCESS;
    }
    
    auto instance_data = LayerData::getInstance().getInstance(VK_NULL_HANDLE);
    if (instance_data) {
        return instance_data->dispatch_table.EnumerateDeviceExtensionProperties(
            physicalDevice, pLayerName, pPropertyCount, pProperties);
    }
    
    return VK_ERROR_INITIALIZATION_FAILED;
}

// GetProcAddr functions
VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(
    VkInstance instance,
    const char* pName)
{
    #define INTERCEPT(name) \
        if (strcmp(pName, #name) == 0) return (PFN_vkVoidFunction)name
    
    INTERCEPT(vkCreateInstance);
    INTERCEPT(vkDestroyInstance);
    INTERCEPT(vkGetInstanceProcAddr);
    INTERCEPT(vkGetDeviceProcAddr);
    INTERCEPT(vkEnumerateInstanceLayerProperties);
    INTERCEPT(vkEnumerateDeviceLayerProperties);
    INTERCEPT(vkEnumerateInstanceExtensionProperties);
    INTERCEPT(vkEnumerateDeviceExtensionProperties);
    INTERCEPT(vkCreateDevice);
    
    #undef INTERCEPT
    
    if (instance == VK_NULL_HANDLE) {
        return nullptr;
    }
    
    auto instance_data = LayerData::getInstance().getInstance(instance);
    if (!instance_data) {
        return nullptr;
    }
    
    return instance_data->dispatch_table.GetInstanceProcAddr(instance, pName);
}

VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(
    VkDevice device,
    const char* pName)
{
    #define INTERCEPT(name) \
        if (strcmp(pName, #name) == 0) return (PFN_vkVoidFunction)name
    
    INTERCEPT(vkGetDeviceProcAddr);
    INTERCEPT(vkDestroyDevice);
    INTERCEPT(vkCreateSwapchainKHR);
    INTERCEPT(vkDestroySwapchainKHR);
    INTERCEPT(vkQueuePresentKHR);
    
    #undef INTERCEPT
    
    auto device_data = LayerData::getInstance().getDevice(device);
    if (!device_data) {
        return nullptr;
    }
    
    return device_data->dispatch_table.GetDeviceProcAddr(device, pName);
}

} // extern "C"

// Helper function implementations
namespace layer_utils {

uint32_t findMemoryType(const VkPhysicalDeviceMemoryProperties& mem_props,
                       uint32_t type_filter, VkMemoryPropertyFlags properties) {
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && 
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

VkImageView createImageView(VkDevice device, VkImage image, VkFormat format,
                           VkImageAspectFlags aspect_flags) {
    auto device_data = LayerData::getInstance().getDevice(device);
    if (!device_data) return VK_NULL_HANDLE;
    
    VkImageViewCreateInfo view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.subresourceRange.aspectMask = aspect_flags;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    
    VkImageView view;
    device_data->dispatch_table.CreateImageView(device, &view_info, nullptr, &view);
    return view;
}

VkSemaphore createSemaphore(VkDevice device) {
    auto device_data = LayerData::getInstance().getDevice(device);
    if (!device_data) return VK_NULL_HANDLE;
    
    VkSemaphoreCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    
    VkSemaphore semaphore;
    device_data->dispatch_table.CreateSemaphore(device, &info, nullptr, &semaphore);
    return semaphore;
}

VkFence createFence(VkDevice device, bool signaled) {
    auto device_data = LayerData::getInstance().getDevice(device);
    if (!device_data) return VK_NULL_HANDLE;
    
    VkFenceCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (signaled) {
        info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    }
    
    VkFence fence;
    device_data->dispatch_table.CreateFence(device, &info, nullptr, &fence);
    return fence;
}

VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool pool) {
    auto device_data = LayerData::getInstance().getDevice(device);
    if (!device_data) return VK_NULL_HANDLE;
    
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandPool = pool;
    alloc_info.commandBufferCount = 1;
    
    VkCommandBuffer cmd_buffer;
    device_data->dispatch_table.AllocateCommandBuffers(device, &alloc_info, &cmd_buffer);
    
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    device_data->dispatch_table.BeginCommandBuffer(cmd_buffer, &begin_info);
    return cmd_buffer;
}

void endSingleTimeCommands(VkDevice device, VkCommandPool pool,
                          VkQueue queue, VkCommandBuffer cmd_buffer) {
    auto device_data = LayerData::getInstance().getDevice(device);
    if (!device_data) return;
    
    device_data->dispatch_table.EndCommandBuffer(cmd_buffer);
    
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buffer;
    
    device_data->dispatch_table.QueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
    device_data->dispatch_table.QueueWaitIdle(queue);
    
    device_data->dispatch_table.FreeCommandBuffers(device, pool, 1, &cmd_buffer);
}

void transitionImageLayout(VkCommandBuffer cmd, VkImage image,
                          VkImageLayout old_layout, VkImageLayout new_layout,
                          VkImageSubresourceRange subresource_range,
                          VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    // Find device data from command buffer (simplified approach)
    DeviceData* device_data = nullptr;
    for (auto& [device, data] : LayerData::getInstance().devices) {
        device_data = data.get();
        break;
    }
    if (!device_data) return;
    
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = subresource_range;
    
    // Source access mask
    switch (old_layout) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
        barrier.srcAccessMask = 0;
        break;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        break;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        break;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        break;
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        break;
    default:
        barrier.srcAccessMask = 0;
        break;
    }
    
    // Destination access mask
    switch (new_layout) {
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        break;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        break;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        break;
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        break;
    case VK_IMAGE_LAYOUT_GENERAL:
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        break;
    default:
        barrier.dstAccessMask = 0;
        break;
    }
    
    device_data->dispatch_table.CmdPipelineBarrier(
        cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

} // namespace layer_utils
