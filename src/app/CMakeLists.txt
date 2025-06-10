SET(APP_DIR ${CMAKE_CURRENT_LIST_DIR})
cmake_path(GET APP_DIR PARENT_PATH APP_PARENT_DIR)

SET(APP_LIBS    spdlog::spdlog 
                glm::glm 
                OpenGL 
                EGL
                glad
                glfw
                tinyobjloader
                CUDA::cudart_static 
                CUDA::nvrtc
                OptiX)

get_target_property(OPTIX_INCLUDE_DIR OptiX INCLUDE_DIRECTORIES)

add_executable(OptiXApp main.cu
                        window.cpp
                        camera.cpp
                        gl_renderer.cpp
                        ssao.cpp
                        optix_renderer.cu)

target_compile_definitions(OptiXApp PUBLIC SHADER_DIR="${APP_DIR}/shader/")
target_compile_definitions(OptiXApp PUBLIC MODULE_DIR="${APP_DIR}/modules/")
target_compile_definitions(OptiXApp PUBLIC OPTIX_INCLUDE_DIR="${OPTIX_INCLUDE_DIR}/")
target_compile_definitions(OptiXApp PUBLIC CUDA_TOOLKIT_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}/")

target_include_directories(OptiXApp PUBLIC ${APP_PARENT_DIR} ${CUDAToolkit_INCLUDE_DIRS} ${APP_DIR}/modules/)
target_link_libraries(OptiXApp PUBLIC ${APP_LIBS})