SET(PYBINDMWIR_DIR ${CMAKE_CURRENT_LIST_DIR})
cmake_path(GET PYBINDMWIR_DIR PARENT_PATH PYBINDMWIR_PARENT_DIR)



pybind11_add_module(PyBindMWIR  pybindmwir.cpp
                                antenna.cpp
                                many_worlds.cpp
                                mesh.cpp
                                forward_renderer.cpp
                                inverse_renderer.cpp
                                scene.cpp
                                signal.cpp)
target_include_directories(PyBindMWIR PUBLIC ${PYBINDMWIR_PARENT_DIR} ${Python3_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(PyBindMWIR PUBLIC MWIR ${Python3_LIBRARIES} ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
set_target_properties(PyBindMWIR PROPERTIES POSITION_INDEPENDENT_CODE ON)











