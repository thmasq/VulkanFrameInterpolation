#pragma once

#include <vulkan/vulkan.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <condition_variable>

// Forward declarations
struct DeviceData;

class FrameInterpolator {
public:
    explicit FrameInterpolator(DeviceData* device_data);
    ~FrameInterpolator();
    
    // Swapchain management
    void initSwapchain(VkSwapchainKHR swapchain, VkFormat format, 
                      VkExtent2D extent, uint32_t image_count);
    void cleanupSwapchain(VkSwapchainKHR swapchain);
    
    // Frame interpolation request
    struct InterpolationRequest {
        VkSwapchainKHR swapchain;
        VkImage prev_frame;
        VkImageView prev_frame_view;
        VkImage curr_frame;
        VkImageView curr_frame_view;
        VkImage output_image;
        uint64_t frame_number;
        float interpolation_factor;
    };
    
    // Main interpolation function
    VkImage interpolateFrame(const InterpolationRequest& request);
    
private:
    DeviceData* device_data_;
    
    // Pipeline data
    struct ComputePipeline {
        VkPipeline pipeline;
        VkPipelineLayout layout;
        VkDescriptorSetLayout desc_set_layout;
        VkDescriptorPool desc_pool;
    };
    
    // Pipelines for different passes
    ComputePipeline optical_flow_pipeline_;
    ComputePipeline interpolation_pipeline_;
    ComputePipeline motion_estimation_pipeline_;
    
    // Shader modules
    VkShaderModule optical_flow_shader_;
    VkShaderModule interpolation_shader_;
    VkShaderModule motion_estimation_shader_;
    
    // Per-swapchain resources
    struct SwapchainResources {
        VkFormat format;
        VkExtent2D extent;
        uint32_t image_count;
        
        // Intermediate buffers
        struct FrameBuffer {
            VkImage image;
            VkImageView view;
            VkDeviceMemory memory;
        };
        
        // Motion vector buffers (2 components, R16G16_SFLOAT)
        std::vector<FrameBuffer> motion_vector_buffers;
        
        // Optical flow buffers
        std::vector<FrameBuffer> optical_flow_buffers;
        
        // Interpolated frame buffers
        std::vector<FrameBuffer> interpolated_buffers;
        
        // Descriptor sets
        std::vector<VkDescriptorSet> optical_flow_desc_sets;
        std::vector<VkDescriptorSet> interpolation_desc_sets;
        std::vector<VkDescriptorSet> motion_estimation_desc_sets;
        
        // Command buffers
        std::vector<VkCommandBuffer> command_buffers;
        
        // Current buffer index
        uint32_t current_buffer_index = 0;
    };
    
    std::unordered_map<VkSwapchainKHR, std::unique_ptr<SwapchainResources>> swapchain_resources_;
    std::mutex resources_mutex_;
    
    // Async compute thread
    std::thread compute_thread_;
    std::queue<InterpolationRequest> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    bool should_stop_ = false;
    
    // Constants buffer
    struct InterpolationConstants {
        float motion_scale[2];
        float interpolation_factor;
        float depth_threshold;
        float motion_threshold;
        float disocclusion_threshold;
        uint32_t frame_width;
        uint32_t frame_height;
        uint32_t block_size;
        uint32_t search_radius;
        uint32_t quality_level;
        uint32_t _padding;
    };
    
    VkBuffer constants_buffer_;
    VkDeviceMemory constants_memory_;
    
    // Helper functions
    void createPipelines();
    void createShaderModules();
    void createConstantsBuffer();
    
    VkShaderModule loadShaderModule(const std::string& path);
    VkShaderModule loadEmbeddedShaderModule(const char* shader_name);  // Added this declaration
    
    void createFrameBuffer(SwapchainResources::FrameBuffer& buffer,
                          VkFormat format, VkExtent2D extent,
                          VkImageUsageFlags usage);
    void destroyFrameBuffer(SwapchainResources::FrameBuffer& buffer);
    
    void createDescriptorSets(SwapchainResources& resources);
    void updateDescriptorSets(SwapchainResources& resources, uint32_t buffer_index,
                            const InterpolationRequest& request);
    
    void computeThreadFunc();
    void processInterpolationRequest(const InterpolationRequest& request);
    
    // FidelityFX integration
    void initializeFidelityFX();
    void shutdownFidelityFX();
    
    // Optical flow using AMD FidelityFX
    void computeOpticalFlow(VkCommandBuffer cmd, 
                           VkImageView prev_frame, VkImageView curr_frame,
                           VkImageView output_flow,
                           VkExtent2D extent);
    
    // Frame interpolation
    void interpolateFrameCompute(VkCommandBuffer cmd,
                                VkImageView prev_frame, VkImageView curr_frame,
                                VkImageView optical_flow, VkImageView motion_vectors,
                                VkImageView output_frame,
                                float interpolation_factor,
                                VkExtent2D extent);
};
