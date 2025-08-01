# Shader compilation

# Find Vulkan shader compiler
find_program(GLSLC glslc HINTS 
    $ENV{VULKAN_SDK}/bin
    /usr/bin
    /usr/local/bin
)

if(NOT GLSLC)
    message(WARNING "glslc not found, shaders will not be compiled")
    return()
endif()

# Shader source files
set(SHADER_SOURCES
    interpolate_frame.comp
    motion_estimation.comp
    optical_flow.comp
)

# Output directory for compiled shaders
set(SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Custom target to compile all shaders
add_custom_target(shaders ALL)

# Compile each shader
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER})
    set(SHADER_OUTPUT ${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv)
    
    add_custom_command(
        OUTPUT ${SHADER_OUTPUT}
        COMMAND ${GLSLC} 
                -fshader-stage=compute
                -O
                -g
                -o ${SHADER_OUTPUT}
                ${SHADER_SOURCE}
        DEPENDS ${SHADER_SOURCE}
        COMMENT "Compiling shader ${SHADER}"
    )
    
    add_custom_target(${SHADER_NAME}_shader DEPENDS ${SHADER_OUTPUT})
    add_dependencies(shaders ${SHADER_NAME}_shader)
endforeach()

# Create the optical flow shader (placeholder for now)
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/optical_flow.comp
"#version 450

// Placeholder for optical flow computation
// This would implement AMD FidelityFX Optical Flow algorithm

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D previousFrame;
layout(set = 0, binding = 1) uniform sampler2D currentFrame;
layout(set = 0, binding = 2, rg16f) uniform writeonly image2D opticalFlowField;

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    
    // Placeholder: simple frame difference
    vec2 uv = (vec2(coord) + 0.5) / vec2(imageSize(opticalFlowField));
    vec3 prev = texture(previousFrame, uv).rgb;
    vec3 curr = texture(currentFrame, uv).rgb;
    
    vec2 flow = (curr.xy - prev.xy) * 0.1;
    imageStore(opticalFlowField, coord, vec4(flow, 0.0, 0.0));
}
")

# Install compiled shaders
install(FILES 
    ${SHADER_OUTPUT_DIR}/interpolate_frame.spv
    ${SHADER_OUTPUT_DIR}/motion_estimation.spv
    ${SHADER_OUTPUT_DIR}/optical_flow.spv
    DESTINATION ${CMAKE_INSTALL_PREFIX}/share/vulkan/shaders/frame_interpolation
)

# Copy shaders to binary directory for development
add_custom_command(TARGET shaders POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${SHADER_OUTPUT_DIR}/*.spv
            ${CMAKE_BINARY_DIR}/shaders
    COMMENT "Copying shaders to binary directory"
)
