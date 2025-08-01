# Shader compilation with C header generation

# Find required tools
find_program(GLSLC glslc HINTS 
    $ENV{VULKAN_SDK}/bin
    /usr/bin
    /usr/local/bin
)

find_program(XXD xxd HINTS
    /usr/bin
    /bin
)

if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found - required for shader compilation")
endif()

if(NOT XXD)
    message(FATAL_ERROR "xxd not found - required for C header generation")
endif()

# Shader source files
set(SHADER_SOURCES
    interpolate_frame.comp
    motion_estimation.comp
    optical_flow.comp
)

# Output directories
set(SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(HEADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/include/shaders)

# Create output directories
file(MAKE_DIRECTORY ${HEADER_OUTPUT_DIR})

# Custom target for all shaders
add_custom_target(shaders ALL)

# Lists to collect generated files
set(GENERATED_HEADERS "")
set(GENERATED_SPV_FILES "")

# Process each shader
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    
    set(SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER})
    set(SHADER_SPV ${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv)
    set(SHADER_HEADER ${HEADER_OUTPUT_DIR}/${SHADER_NAME}.h)
    
    # Compile GLSL to SPIR-V
    add_custom_command(
        OUTPUT ${SHADER_SPV}
        COMMAND ${GLSLC} 
                -fshader-stage=compute
                -O
                -g
                --target-env=vulkan1.3
                -o ${SHADER_SPV}
                ${SHADER_SOURCE}
        DEPENDS ${SHADER_SOURCE}
        COMMENT "Compiling shader ${SHADER} to SPIR-V"
    )
    
    # Convert SPIR-V to C header using xxd
    add_custom_command(
        OUTPUT ${SHADER_HEADER}
        COMMAND ${XXD} -i ${SHADER_NAME}.spv > ${SHADER_HEADER}
        WORKING_DIRECTORY ${SHADER_OUTPUT_DIR}
        DEPENDS ${SHADER_SPV}
        COMMENT "Generating C header for ${SHADER_NAME}"
    )
    
    # Create individual target for this shader
    add_custom_target(${SHADER_NAME}_shader DEPENDS ${SHADER_HEADER})
    add_dependencies(shaders ${SHADER_NAME}_shader)
    
    list(APPEND GENERATED_HEADERS ${SHADER_HEADER})
    list(APPEND GENERATED_SPV_FILES ${SHADER_SPV})
endforeach()

# Generate the main shader headers include file
set(SHADERS_INCLUDE_FILE ${HEADER_OUTPUT_DIR}/embedded_shaders.h)
add_custom_command(
    OUTPUT ${SHADERS_INCLUDE_FILE}
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/generate_shaders_header.cmake
    DEPENDS ${GENERATED_HEADERS}
    COMMENT "Generating main shaders include file"
)

add_custom_target(shaders_header DEPENDS ${SHADERS_INCLUDE_FILE})
add_dependencies(shaders shaders_header)

# Export variables for parent scope  
set(SHADER_HEADERS_DIR ${HEADER_OUTPUT_DIR} PARENT_SCOPE)
set(GENERATED_SHADER_HEADERS ${GENERATED_HEADERS} PARENT_SCOPE)

# Copy development shaders (optional, for debugging)
add_custom_command(TARGET shaders POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${GENERATED_SPV_FILES}
            ${CMAKE_BINARY_DIR}/shaders/
    COMMENT "Copying SPIR-V files to binary directory"
)

# Install shader headers (for external projects)
install(FILES ${GENERATED_HEADERS} ${SHADERS_INCLUDE_FILE}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/vulkan_frame_interpolation/shaders)
