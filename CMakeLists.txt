cmake_minimum_required(VERSION 3.16)
project(VulkanFrameInterpolation VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(ENABLE_VALIDATION "Enable Vulkan validation layers" ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add external dependencies
set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)
set(FFX_SDK_DIR ${EXTERNAL_DIR}/FidelityFX)

# Global include directories
include_directories(
    ${Vulkan_INCLUDE_DIRS}
    ${FFX_SDK_DIR}/include
    ${CMAKE_SOURCE_DIR}/layer
    ${CMAKE_SOURCE_DIR}/include
)

# Global definitions
add_definitions(-DVULKAN_HPP_NO_EXCEPTIONS)
if(UNIX)
    add_definitions(-D_GNU_SOURCE)
endif()

# Add subdirectories
add_subdirectory(shaders)
add_subdirectory(layer)
add_subdirectory(launcher)

# Install layer manifest
install(FILES ${CMAKE_SOURCE_DIR}/layer/VkLayer_frame_interpolation.json
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/vulkan/implicit_layer.d)

# Create uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
