# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#     if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
#         set(CMAKE_CUDA_ARCHITECTURES "native")
#     else()
#         set(CMAKE_CUDA_ARCHITECTURES 70)
#         message(STATUS "CMake < 3.18 detected, using default architecture 70")
#     endif()
# endif()
# message(STATUS "Compiling CUDA code for architecture: ${CMAKE_CUDA_ARCHITECTURES}")








find_package(spdlog REQUIRED)
find_package(glm REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(X11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED CONFIG)

if(NOT TARGET CUDA::nvToolsExt AND TARGET CUDA::nvtx3)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    target_compile_definitions(
        CUDA::nvToolsExt INTERFACE
        TORCH_CUDA_USE_NVTX3
    )
    target_link_libraries(CUDA::nvToolsExt INTERFACE CUDA::nvtx3)
endif()
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
set(CMAKE_CUDA_ARCHITECTURES 89)