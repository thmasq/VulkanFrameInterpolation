# Launcher executable
add_executable(vk-frame-interpolation main.cpp)

target_link_libraries(vk-frame-interpolation
    PRIVATE
        ${CMAKE_DL_LIBS}
        Threads::Threads
)

target_compile_features(vk-frame-interpolation PRIVATE cxx_std_17)

# Set RPATH for finding the layer library
set_target_properties(vk-frame-interpolation PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Install the launcher
install(TARGETS vk-frame-interpolation
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)


# Create the wrapper script template
file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/vk-frame-interpolation.sh.in"
"#!/bin/bash
# Vulkan Frame Interpolation Launcher Wrapper
exec \"@CMAKE_INSTALL_PREFIX@/bin/vk-frame-interpolation\" \"$@\"
")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vk-frame-interpolation.sh"
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        RENAME vkfi)
