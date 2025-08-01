#pragma once

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#include <vulkan/vk_layer_dispatch_table.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <atomic>

// Forward declarations
class FrameInterpolator;

// Layer name
#define LAYER_NAME "VK_LAYER_frame_interpolation"
#define LAYER_DESCRIPTION "Frame interpolation layer for doubling framerate"

// Per-instance data
struct InstanceData {
    VkLayerInstanceDispatchTable dispatch_table;
    VkInstance instance;
    bool debug_enabled;
};

// Per-device data
struct DeviceData {
    VkLayerDispatchTable dispatch_table;
    VkDevice device;
    VkPhysicalDevice physical_device;
    InstanceData* instance_data;
    
    // Device properties
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceMemoryProperties memory_properties;
    
    // Queue families
    uint32_t graphics_queue_family;
    uint32_t compute_queue_family;
    uint32_t present_queue_family;
    
    // Compute queue for async frame generation
    VkQueue compute_queue;
    VkCommandPool compute_command_pool;
    
    // Frame interpolator
    std::unique_ptr<FrameInterpolator> interpolator;
};

// Per-swapchain data
struct SwapchainData {
    VkSwapchainKHR swapchain;
    DeviceData* device_data;
    
    // Swapchain properties
    VkFormat format;
    VkExtent2D extent;
    uint32_t image_count;
    std::vector<VkImage> images;
    std::vector<VkImageView> image_views;
    
    // Synchronization
    std::vector<VkSemaphore> interpolation_complete_semaphores;
    std::vector<VkFence> interpolation_fences;
    
    // Frame tracking
    std::atomic<uint64_t> frame_number{0};
    std::atomic<bool> interpolation_active{false};
    
    // Circular buffer for frame history
    static constexpr uint32_t FRAME_HISTORY_SIZE = 3;
    struct FrameData {
        VkImage image;
        VkImageView view;
        VkSemaphore ready_semaphore;
        uint64_t frame_number;
        bool valid;
    };
    std::array<FrameData, FRAME_HISTORY_SIZE> frame_history;
    uint32_t history_write_index;
    
    // Performance metrics
    struct {
        std::atomic<uint64_t> frames_presented{0};
        std::atomic<uint64_t> frames_interpolated{0};
        std::atomic<uint64_t> interpolation_time_us{0};
    } stats;
};

// Global layer data
class LayerData {
public:
    static LayerData& getInstance() {
        static LayerData instance;
        return instance;
    }
    
    // Instance management
    void addInstance(VkInstance instance, InstanceData* data);
    void removeInstance(VkInstance instance);
    InstanceData* getInstance(VkInstance instance);
    
    // Device management
    void addDevice(VkDevice device, DeviceData* data);
    void removeDevice(VkDevice device);
    DeviceData* getDevice(VkDevice device);
    
    // Swapchain management
    void addSwapchain(VkSwapchainKHR swapchain, SwapchainData* data);
    void removeSwapchain(VkSwapchainKHR swapchain);
    SwapchainData* getSwapchain(VkSwapchainKHR swapchain);
    
    // Layer settings
    struct Settings {
        bool enabled = true;
        int target_fps = 0;  // 0 = 2x input
        int quality = 1;     // 0=fast, 1=balanced, 2=quality
        bool debug = false;
        bool show_stats = false;
    } settings;
    
    void loadSettings();
    
private:
    LayerData();
    ~LayerData() = default;
    
    std::mutex instance_mutex;
    std::mutex device_mutex;
    std::mutex swapchain_mutex;
    
    std::unordered_map<VkInstance, std::unique_ptr<InstanceData>> instances;
    std::unordered_map<VkDevice, std::unique_ptr<DeviceData>> devices;
    std::unordered_map<VkSwapchainKHR, std::unique_ptr<SwapchainData>> swapchains;
};

// Helper functions
namespace layer_utils {
    // Memory allocation helpers
    uint32_t findMemoryType(const VkPhysicalDeviceMemoryProperties& mem_props,
                           uint32_t type_filter, VkMemoryPropertyFlags properties);
    
    // Image helpers
    VkImageView createImageView(VkDevice device, VkImage image, VkFormat format,
                               VkImageAspectFlags aspect_flags);
    
    // Synchronization helpers
    VkSemaphore createSemaphore(VkDevice device);
    VkFence createFence(VkDevice device, bool signaled = false);
    
    // Command buffer helpers
    VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool pool);
    void endSingleTimeCommands(VkDevice device, VkCommandPool pool,
                              VkQueue queue, VkCommandBuffer cmd_buffer);
    
    // Transition helpers
    void transitionImageLayout(VkCommandBuffer cmd, VkImage image,
                              VkImageLayout old_layout, VkImageLayout new_layout,
                              VkImageSubresourceRange subresource_range,
                              VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage);
}
