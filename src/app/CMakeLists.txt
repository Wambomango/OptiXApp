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

SET(APP_INCLUDES ${APP_PARENT_DIR} ${CUDAToolkit_INCLUDE_DIRS})
SET(SHADER_DIR ${APP_DIR}/shader)


add_executable(OptiXApp main.cu
                        gl_renderer.cpp
                        optix_renderer.cu)

target_compile_definitions(OptiXApp PUBLIC SHADER_DIR="${SHADER_DIR}/")
target_include_directories(OptiXApp PUBLIC ${APP_INCLUDES})
target_link_libraries(OptiXApp PUBLIC ${APP_LIBS})