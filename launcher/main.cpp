#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>
#include <filesystem>

namespace fs = std::filesystem;

// ANSI color codes for terminal output
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"

class VulkanLayerLauncher {
private:
    std::string layer_path;
    std::string layer_name = "VK_LAYER_frame_interpolation";
    
public:
    VulkanLayerLauncher() {
        // Try to find the layer library
        std::vector<std::string> search_paths = {
            "./lib/libVkLayer_frame_interpolation.so",
            "/usr/local/lib/libVkLayer_frame_interpolation.so",
            "/usr/lib/libVkLayer_frame_interpolation.so",
            "../lib/libVkLayer_frame_interpolation.so",
            "./build/lib/libVkLayer_frame_interpolation.so"
        };
        
        for (const auto& path : search_paths) {
            if (fs::exists(path)) {
                layer_path = fs::absolute(path).string();
                break;
            }
        }
        
        if (layer_path.empty()) {
            throw std::runtime_error("Could not find VkLayer_frame_interpolation.so");
        }
    }
    
    void print_usage(const char* program_name) {
        std::cout << BLUE << "Vulkan Frame Interpolation Launcher" << RESET << "\n\n";
        std::cout << "Usage: " << program_name << " [options] <program> [program arguments]\n\n";
        std::cout << "Options:\n";
        std::cout << "  -h, --help           Show this help message\n";
        std::cout << "  -v, --verbose        Enable verbose output\n";
        std::cout << "  -d, --debug          Enable Vulkan validation layers\n";
        std::cout << "  -f, --fps <target>   Set target FPS (default: 2x input)\n";
        std::cout << "  -q, --quality <0-2>  Set quality (0=fast, 1=balanced, 2=quality)\n";
        std::cout << "\n";
        std::cout << "Examples:\n";
        std::cout << "  " << program_name << " /usr/bin/vkcube\n";
        std::cout << "  " << program_name << " -v -f 120 /path/to/game\n";
        std::cout << "\n";
    }
    
    int launch(int argc, char* argv[]) {
        bool verbose = false;
        bool debug = false;
        int target_fps = 0;
        int quality = 1;
        
        // Parse command line arguments
        int i = 1;
        for (; i < argc; ++i) {
            std::string arg = argv[i];
            
            if (arg == "-h" || arg == "--help") {
                print_usage(argv[0]);
                return 0;
            } else if (arg == "-v" || arg == "--verbose") {
                verbose = true;
            } else if (arg == "-d" || arg == "--debug") {
                debug = true;
            } else if (arg == "-f" || arg == "--fps") {
                if (++i >= argc) {
                    std::cerr << RED << "Error: --fps requires an argument" << RESET << "\n";
                    return 1;
                }
                target_fps = std::atoi(argv[i]);
            } else if (arg == "-q" || arg == "--quality") {
                if (++i >= argc) {
                    std::cerr << RED << "Error: --quality requires an argument" << RESET << "\n";
                    return 1;
                }
                quality = std::atoi(argv[i]);
                if (quality < 0 || quality > 2) {
                    std::cerr << RED << "Error: quality must be 0, 1, or 2" << RESET << "\n";
                    return 1;
                }
            } else if (arg[0] != '-') {
                break; // Found the program to launch
            } else {
                std::cerr << RED << "Error: Unknown option " << arg << RESET << "\n";
                return 1;
            }
        }
        
        if (i >= argc) {
            std::cerr << RED << "Error: No program specified" << RESET << "\n";
            print_usage(argv[0]);
            return 1;
        }
        
        const char* program = argv[i];
        
        // Check if program exists
        if (!fs::exists(program)) {
            std::cerr << RED << "Error: Program not found: " << program << RESET << "\n";
            return 1;
        }
        
        if (verbose) {
            std::cout << GREEN << "Frame Interpolation Layer Configuration:" << RESET << "\n";
            std::cout << "  Layer path: " << layer_path << "\n";
            std::cout << "  Target FPS: " << (target_fps > 0 ? std::to_string(target_fps) : "2x input") << "\n";
            std::cout << "  Quality: " << (quality == 0 ? "Fast" : quality == 1 ? "Balanced" : "Quality") << "\n";
            std::cout << "  Program: " << program << "\n";
            std::cout << "\n";
        }
        
        // Set up environment variables
        std::vector<std::string> env_vars;
        
        // Add our layer to VK_INSTANCE_LAYERS
        const char* existing_layers = std::getenv("VK_INSTANCE_LAYERS");
        std::string layers = layer_name;
        if (existing_layers && strlen(existing_layers) > 0) {
            layers = std::string(existing_layers) + ":" + layer_name;
        }
        env_vars.push_back("VK_INSTANCE_LAYERS=" + layers);
        
        // Add layer path to VK_LAYER_PATH
        const char* existing_layer_path = std::getenv("VK_LAYER_PATH");
        std::string layer_dir = fs::path(layer_path).parent_path().string();
        std::string vk_layer_path = layer_dir;
        if (existing_layer_path && strlen(existing_layer_path) > 0) {
            vk_layer_path = std::string(existing_layer_path) + ":" + layer_dir;
        }
        env_vars.push_back("VK_LAYER_PATH=" + vk_layer_path);
        
        // Add configuration environment variables
        env_vars.push_back("VK_FRAME_INTERPOLATION_ENABLED=1");
        if (target_fps > 0) {
            env_vars.push_back("VK_FRAME_INTERPOLATION_TARGET_FPS=" + std::to_string(target_fps));
        }
        env_vars.push_back("VK_FRAME_INTERPOLATION_QUALITY=" + std::to_string(quality));
        
        if (debug) {
            env_vars.push_back("VK_LOADER_DEBUG=all");
            env_vars.push_back("VK_FRAME_INTERPOLATION_DEBUG=1");
            
            // Add validation layers
            const char* validation_layers = "VK_LAYER_KHRONOS_validation";
            if (existing_layers && strstr(existing_layers, "VK_LAYER_KHRONOS_validation") == nullptr) {
                env_vars[0] = "VK_INSTANCE_LAYERS=" + std::string(validation_layers) + ":" + layers;
            }
        }
        
        if (verbose) {
            std::cout << YELLOW << "Environment variables:" << RESET << "\n";
            for (const auto& var : env_vars) {
                std::cout << "  " << var << "\n";
            }
            std::cout << "\n";
        }
        
        // Fork and execute the program
        pid_t pid = fork();
        if (pid == -1) {
            std::cerr << RED << "Error: Failed to fork process" << RESET << "\n";
            return 1;
        }
        
        if (pid == 0) {
            // Child process
            
            // Set environment variables
            for (const auto& var : env_vars) {
                size_t eq_pos = var.find('=');
                if (eq_pos != std::string::npos) {
                    std::string key = var.substr(0, eq_pos);
                    std::string value = var.substr(eq_pos + 1);
                    setenv(key.c_str(), value.c_str(), 1);
                }
            }
            
            // Prepare arguments for execv
            std::vector<char*> exec_args;
            for (int j = i; j < argc; ++j) {
                exec_args.push_back(argv[j]);
            }
            exec_args.push_back(nullptr);
            
            // Execute the program
            execvp(program, exec_args.data());
            
            // If we get here, execvp failed
            std::cerr << RED << "Error: Failed to execute " << program << ": " 
                      << strerror(errno) << RESET << "\n";
            exit(1);
        } else {
            // Parent process
            if (verbose) {
                std::cout << GREEN << "Launched process with PID: " << pid << RESET << "\n\n";
            }
            
            // Wait for child process to finish
            int status;
            waitpid(pid, &status, 0);
            
            if (WIFEXITED(status)) {
                int exit_code = WEXITSTATUS(status);
                if (verbose) {
                    std::cout << "\n" << BLUE << "Process exited with code: " 
                              << exit_code << RESET << "\n";
                }
                return exit_code;
            } else if (WIFSIGNALED(status)) {
                int sig = WTERMSIG(status);
                std::cerr << "\n" << RED << "Process terminated by signal: " 
                          << sig << RESET << "\n";
                return 128 + sig;
            }
        }
        
        return 0;
    }
};

int main(int argc, char* argv[]) {
    try {
        VulkanLayerLauncher launcher;
        return launcher.launch(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << RED << "Error: " << e.what() << RESET << "\n";
        return 1;
    }
}
