cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cu LANGUAGES CXX CUDA)

# ===========================================================
# Project Configuration
# ===========================================================

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable CUDA language
enable_language(CUDA)

# Set CUDA standard (optional, matches C++ standard)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ===========================================================
# Options
# ===========================================================

# Option to build with cuDNN support
option(USE_CUDNN "Enable cuDNN support" OFF)

# Option to build with CUTLASS support
option(USE_CUTLASS "Enable CUTLASS support" ON)

# ===========================================================
# Find Packages
# ===========================================================

# Find CUDA Toolkit
find_package(CUDA REQUIRED)

# Find cuDNN if enabled
if(USE_CUDNN)
    find_package(CUDNN REQUIRED)
    if(NOT CUDNN_FOUND)
        message(FATAL_ERROR "cuDNN not found. Please install cuDNN or disable USE_CUDNN.")
    endif()
endif()

# Find CUTLASS if enabled
if(USE_CUTLASS)
    # Assuming CUTLASS is located in the 'external/CUTLASS' directory
    set(CUTLASS_DIR "/home/william/Codes/cutlass")
    if(EXISTS "${CUTLASS_DIR}/CMakeLists.txt")
        add_subdirectory(${CUTLASS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/CUTLASS)
        set(CUTLASS_FOUND TRUE)
    else()
        message(FATAL_ERROR "CUTLASS directory not found at ${CUTLASS_DIR}. Please download CUTLASS or disable USE_CUTLASS.")
    endif()
endif()

# ===========================================================
# Include Directories
# ===========================================================

# Include directories for project sources
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Include directories for cuDNN
if(USE_CUDNN)
    include_directories(${CUDNN_INCLUDE_DIRS})
endif()

# Include directories for CUTLASS
if(USE_CUTLASS)
    include_directories(${CUTLASS_INCLUDE_DIRS})
endif()

# ===========================================================
# Source Files
# ===========================================================

# Gather all CUDA source files
file(GLOB CUDA_SOURCES "src/*.cu")

# Optionally, add other source files (e.g., headers, C++ files)
# file(GLOB CPP_SOURCES "src/*.cpp")

# ===========================================================
# Add Executable
# ===========================================================

add_executable(${PROJECT_NAME} ${CUDA_SOURCES})

# ===========================================================
# Link Libraries
# ===========================================================

# Link cuDNN libraries if enabled
if(USE_CUDNN)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDNN_LIBRARIES})
    # On some systems, you might need to link additional CUDA libraries
    # For example:
    # target_link_libraries(bilinear_interp PRIVATE cudart)
endif()

# Link CUTLASS libraries if enabled
if(USE_CUTLASS)
    target_link_libraries(${PROJECT_NAME} PRIVATE cutlass)
endif()

# ===========================================================
# CUDA Properties
# ===========================================================

# Specify CUDA architectures (adjust based on your GPU)
# You can query your GPU's compute capability and set accordingly
# Example for architectures 6.0 (Pascal), 7.0 (Volta), 7.5 (Turing), 8.0 (Ampere)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES "60;61;70;75;80;86"
)

# ===========================================================
# Compiler Flags (Optional)
# ===========================================================

# Enable compiler optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endif()

# ===========================================================
# Post-Build Commands (Optional)
# ===========================================================

# For example, copy necessary runtime libraries or perform other tasks
