cmake_minimum_required(VERSION 3.28)

project(OptiXApp LANGUAGES CUDA C CXX)

set(CMAKE_CXX_STANDARD 17)

include(cmake/handleDeps.cmake)

add_subdirectory(dependencies)
add_subdirectory(src)

