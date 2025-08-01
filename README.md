# Vulkan Frame Interpolation Layer

A Vulkan layer that automatically doubles the framerate of any Vulkan application by generating interpolated frames using AI-based motion estimation and optical flow techniques.

## Features

- **Automatic Frame Doubling**: Intercepts `vkQueuePresentKHR` to generate interpolated frames between real frames
- **AMD FidelityFX Integration**: Uses production-quality optical flow and motion estimation algorithms
- **Asynchronous Processing**: Utilizes compute queues for parallel frame generation
- **Smart Disocclusion Handling**: Detects and handles occlusion/disocclusion artifacts
- **Quality Modes**: Fast, Balanced, and Quality presets for different performance/quality trade-offs
- **Zero Application Modification**: Works with any Vulkan application without code changes

## Requirements

- Linux (tested on Ubuntu 20.04+)
- Vulkan SDK 1.3.0 or newer
- CMake 3.16 or newer
- C++17 compatible compiler
- Vulkan-capable GPU with compute support

## Building

```bash
# Clone the repository
git clone <repository-url>
cd vulkan-frame-interpolation

# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
make -j$(nproc)

# Install (optional)
sudo make install
```

## Usage

### Using the Launcher

The easiest way to use the frame interpolation layer is with the provided launcher:

```bash
# Basic usage
vk-frame-interpolation /path/to/vulkan/application

# With options
vk-frame-interpolation -v -f 120 -q 2 /usr/bin/vkcube

# Show help
vk-frame-interpolation --help
```

### Manual Usage

You can also enable the layer manually using environment variables:

```bash
# Enable the layer
export VK_INSTANCE_LAYERS=VK_LAYER_frame_interpolation
export VK_LAYER_PATH=/usr/local/share/vulkan/implicit_layer.d

# Configure options
export VK_FRAME_INTERPOLATION_ENABLED=1
export VK_FRAME_INTERPOLATION_TARGET_FPS=120  # Optional: default is 2x input
export VK_FRAME_INTERPOLATION_QUALITY=1       # 0=fast, 1=balanced, 2=quality

# Run your application
./your_vulkan_app
```

## Configuration Options

| Environment Variable | Description | Values |
|---------------------|-------------|---------|
| `VK_FRAME_INTERPOLATION_ENABLED` | Enable/disable the layer | 0 or 1 |
| `VK_FRAME_INTERPOLATION_TARGET_FPS` | Target framerate | Integer (0 = 2x input) |
| `VK_FRAME_INTERPOLATION_QUALITY` | Quality preset | 0, 1, or 2 |
| `VK_FRAME_INTERPOLATION_DEBUG` | Enable debug output | 0 or 1 |
| `VK_FRAME_INTERPOLATION_SHOW_STATS` | Show performance statistics | 0 or 1 |

## How It Works

### Core Pipeline

1. **Frame Capture**: Intercepts frames at `vkQueuePresentKHR`
2. **Motion Estimation**: Uses 8x8 block matching with SAD (Sum of Absolute Differences)
3. **Optical Flow**: Computes per-pixel motion vectors using AMD FidelityFX algorithms
4. **Frame Interpolation**: Generates intermediate frames using motion-compensated interpolation
5. **Presentation**: Alternates between real and generated frames

### Technical Details

- **Circular Frame Buffer**: Maintains 3-frame history for temporal stability
- **Async Compute**: Parallel processing on dedicated compute queue
- **GPU-Resident**: All processing happens on GPU for <8ms latency
- **Disocclusion Detection**: Handles revealed areas using depth-aware blending

## Project Structure

```
vulkan-frame-interpolation/
├── CMakeLists.txt              # Root build configuration
├── README.md                   # This file
├── launcher/
│   ├── main.cpp               # Launcher application
│   └── CMakeLists.txt
├── layer/
│   ├── vk_layer_frame_interpolation.cpp    # Main layer implementation
│   ├── vk_layer_frame_interpolation.h
│   ├── frame_interpolator.cpp              # Frame interpolation logic
│   ├── frame_interpolator.h
│   ├── VkLayer_frame_interpolation.json   # Layer manifest
│   └── CMakeLists.txt
├── shaders/
│   ├── interpolate_frame.comp  # Main interpolation shader
│   ├── motion_estimation.comp  # Motion vector computation
│   ├── optical_flow.comp      # Optical flow computation
│   └── CMakeLists.txt
└── external/
    └── FidelityFX/            # AMD FidelityFX SDK files

```

## Integration with FidelityFX

This project integrates AMD FidelityFX SDK components:

1. Copy FidelityFX SDK files to `external/FidelityFX/`
2. The layer uses FidelityFX shaders and algorithms for:
   - Optical flow computation
   - Frame interpolation
   - Motion vector refinement

## Performance

Expected performance impact:
- **Fast mode**: 3-5ms per frame
- **Balanced mode**: 5-8ms per frame  
- **Quality mode**: 8-12ms per frame

Actual performance depends on resolution, GPU, and scene complexity.

## Troubleshooting

### Layer not loading
- Check `VK_LAYER_PATH` includes the layer directory
- Verify the layer JSON manifest is valid
- Run with `VK_LOADER_DEBUG=all` for detailed logs

### Poor interpolation quality
- Try increasing quality level
- Check if the application provides motion vectors
- Some content (UI, text) may not interpolate well

### Performance issues
- Use Fast mode for better performance
- Reduce target framerate
- Check GPU utilization with `nvidia-smi` or `radeontop`

## Known Limitations

- Requires depth buffer access for best results
- UI elements may show artifacts
- Very fast motion can cause ghosting
- Not suitable for competitive gaming (adds latency)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

The AMD FidelityFX SDK components are licensed under their respective licenses.

## Acknowledgments

- AMD for the FidelityFX SDK and optical flow algorithms
- The Vulkan community for documentation and examples
- Contributors and testers
