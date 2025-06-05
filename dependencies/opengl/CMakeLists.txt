add_library(EGL SHARED IMPORTED GLOBAL)
set_target_properties(EGL PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/libEGL.so")
target_include_directories(EGL INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_library(OpenGL SHARED IMPORTED GLOBAL)
set_target_properties(OpenGL PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/libOpenGL.so")
target_include_directories(OpenGL INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")