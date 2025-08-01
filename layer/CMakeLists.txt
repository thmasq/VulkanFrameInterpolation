# Frame Interpolation Vulkan Layer

# Define the layer library
set(LAYER_NAME VkLayer_frame_interpolation)

# Source files
set(LAYER_SOURCES
    vk_layer_frame_interpolation.cpp
    frame_interpolator.cpp
)

# Header files
set(LAYER_HEADERS
    vk_layer_frame_interpolation.h
    frame_interpolator.h
)

# Create the layer library
add_library(${LAYER_NAME} SHARED ${LAYER_SOURCES} ${LAYER_HEADERS})

# Include directories
target_include_directories(${LAYER_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Vulkan_INCLUDE_DIRS}
        ${FFX_SDK_DIR}/include
        ${FFX_SDK_DIR}/include/FidelityFX/gpu
)

# Link libraries
target_link_libraries(${LAYER_NAME}
    PRIVATE
        ${Vulkan_LIBRARIES}
        ${CMAKE_DL_LIBS}
        Threads::Threads
)

# Compile definitions
target_compile_definitions(${LAYER_NAME}
    PRIVATE
        VK_LAYER_EXPORT
        VK_USE_PLATFORM_XCB_KHR  # For Linux/X11
        VK_USE_PLATFORM_WAYLAND_KHR  # For Wayland
)

# Compile options
target_compile_options(${LAYER_NAME}
    PRIVATE
        -Wall
        -Wextra
        -Wno-unused-parameter
        -Wno-missing-field-initializers
        -fPIC
        -fvisibility=hidden
)

# Set properties
set_target_properties(${LAYER_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    PREFIX "lib"
    LIBRARY_OUTPUT_NAME ${LAYER_NAME}
)

# Generate layer JSON manifest with correct paths
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/VkLayer_frame_interpolation.json
    ${CMAKE_CURRENT_BINARY_DIR}/VkLayer_frame_interpolation.json
    COPYONLY
)

# For development, create a JSON that points to the build directory
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/VkLayer_frame_interpolation_dev.json
"{
    \"file_format_version\": \"1.2.0\",
    \"layer\": {
        \"name\": \"VK_LAYER_frame_interpolation\",
        \"type\": \"GLOBAL\",
        \"library_path\": \"${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/lib${LAYER_NAME}.so\",
        \"api_version\": \"1.3.0\",
        \"implementation_version\": \"1\",
        \"description\": \"Frame interpolation layer for doubling framerate (development build)\",
        \"status\": \"STABLE\"
    }
}")

# Install the layer library
install(TARGETS ${LAYER_NAME}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

# Install the layer manifest
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/VkLayer_frame_interpolation.json
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/vulkan/implicit_layer.d)

# Create explicit layer manifest for testing
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/VkLayer_frame_interpolation_explicit.json
"{
    \"file_format_version\": \"1.2.0\",
    \"layer\": {
        \"name\": \"VK_LAYER_frame_interpolation\",
        \"type\": \"INSTANCE\",
        \"library_path\": \"lib${LAYER_NAME}.so\",
        \"api_version\": \"1.3.0\",
        \"implementation_version\": \"1\",
        \"description\": \"Frame interpolation layer for doubling framerate (explicit)\",
        \"status\": \"STABLE\"
    }
}")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/VkLayer_frame_interpolation_explicit.json
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/vulkan/explicit_layer.d
        OPTIONAL)

# Add custom target to validate JSON manifest
find_program(PYTHON_EXECUTABLE python3)
if(PYTHON_EXECUTABLE)
    add_custom_target(validate_layer_json
        COMMAND ${PYTHON_EXECUTABLE} -m json.tool 
                ${CMAKE_CURRENT_SOURCE_DIR}/VkLayer_frame_interpolation.json 
                > /dev/null
        COMMENT "Validating layer JSON manifest"
    )
    add_dependencies(${LAYER_NAME} validate_layer_json)
endif()

# Export compile commands for development
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
