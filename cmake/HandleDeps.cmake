
find_package(spdlog REQUIRED)
find_package(glm REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(X11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED CONFIG)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
