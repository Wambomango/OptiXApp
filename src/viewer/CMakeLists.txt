SET(VIEWER_DIR ${CMAKE_CURRENT_LIST_DIR})
cmake_path(GET VIEWER_DIR PARENT_PATH VIEWER_PARENT_DIR)

SET(APP_LIBS    TBB::tbb 
                spdlog::spdlog 
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

add_executable(Viewer   main.cu
                        window.cpp
                        camera.cpp
                        scene.cpp
                        gl_renderer.cpp
                        ssao.cpp
                        optix_renderer.cu)

target_compile_definitions(Viewer PUBLIC SPDLOG_ACTIVE_LEVEL=5)
target_compile_definitions(Viewer PUBLIC SHADER_DIR="${VIEWER_DIR}/shader/")
target_compile_definitions(Viewer PUBLIC MODULE_DIR="${VIEWER_DIR}/modules/")
target_compile_definitions(Viewer PUBLIC OPTIX_INCLUDE_DIR="${OPTIX_INCLUDE_DIR}/")
target_compile_definitions(Viewer PUBLIC CUDA_TOOLKIT_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}/")

target_include_directories(Viewer PUBLIC ${VIEWER_PARENT_DIR} ${CUDAToolkit_INCLUDE_DIRS} ${VIEWER_DIR}/modules/ ${X11_Xrandr_INCLUDE_PATH} ${TORCH_INCLUDE_DIRS})
target_link_libraries(Viewer PUBLIC ${APP_LIBS} ${TORCH_LIBRARIES})
