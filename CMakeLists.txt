cmake_minimum_required(VERSION 3.18)
project(cuprof LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Header-only library interface
add_library(cuprof INTERFACE)
target_include_directories(cuprof INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Examples
option(CUPROF_BUILD_EXAMPLES "Build cuprof examples" ON)

if(CUPROF_BUILD_EXAMPLES)
    # Basic example
    add_executable(basic_example examples/basic.cu)
    target_link_libraries(basic_example PRIVATE cuprof CUDA::cudart)
    set_target_properties(basic_example PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86;89;90"
    )

    # Warp-specialized example
    add_executable(warp_example examples/warp_specialized.cu)
    target_link_libraries(warp_example PRIVATE cuprof CUDA::cudart)
    set_target_properties(warp_example PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86;89;90"
    )
endif()
