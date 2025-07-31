SET(MWIR_DIR ${CMAKE_CURRENT_LIST_DIR})
cmake_path(GET MWIR_DIR PARENT_PATH MWIR_PARENT_DIR)

SET(MWIR_LIBS   spdlog::spdlog 
                glm::glm 
                CUDA::cudart_static 
                CUDA::nvrtc
                OptiX)

get_target_property(OPTIX_INCLUDE_DIR OptiX INCLUDE_DIRECTORIES)

add_library(MWIR SHARED forward_renderer_impl.cu
                        forward_renderer.cpp
                        forward_pipeline.cpp
                        scene_impl.cpp
                        scene.cpp
                        mesh_impl.cpp
                        mesh.cpp
                        antenna_impl.cpp
                        antenna.cpp
                        signal_impl.cpp
                        signal.cpp)

target_compile_definitions(MWIR PUBLIC MODULE_DIR="${MWIR_DIR}/modules/")
target_compile_definitions(MWIR PUBLIC OPTIX_INCLUDE_DIR="${OPTIX_INCLUDE_DIR}/")
target_compile_definitions(MWIR PUBLIC CUDA_TOOLKIT_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}/")
target_compile_definitions(MWIR PUBLIC SPDLOG_ACTIVE_LEVEL=5)

target_include_directories(MWIR INTERFACE ${MWIR_DIR}/include)
target_include_directories(MWIR PUBLIC ${MWIR_PARENT_DIR} ${CUDAToolkit_INCLUDE_DIRS} ${MWIR_DIR}/modules/ ${TORCH_INCLUDE_DIRS})
target_link_libraries(MWIR PUBLIC ${MWIR_LIBS} ${TORCH_LIBRARIES})

set_target_properties(MWIR PROPERTIES POSITION_INDEPENDENT_CODE ON)











