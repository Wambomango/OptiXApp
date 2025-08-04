mkdir build
cd build

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH="$CUDA_HOME:$CMAKE_PREFIX_PATH"

cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j 8

cd ..
cp -r build/src/pybindmwir/PyBindMWIR.cpython-313-x86_64-linux-gnu.so scripts/PyMWIR/PyBindMWIR.so
cp -r build/src/viewer/Viewer scripts/PyMWIR/Viewer