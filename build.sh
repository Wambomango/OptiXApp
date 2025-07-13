mkdir build
cd build
export CUDA_HOME=$CONDA_PREFIX
export CUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX
export CMAKE_PREFIX_PATH="$CONDA_PREFIX/lib/python3.13/site-packages/torch/share/cmake/Torch:$CONDA_PREFIX:$CMAKE_PREFIX_PATH"
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j 8

cd ..
cp -r build/src/pymwir/PyMWIR.cpython-313-x86_64-linux-gnu.so scripts/PyMWIR.so

# python src/mwir_python/setup.py build
# rm -rf scripts/ManyWorldsInverseRadar.so
# cp -r build/lib.linux-x86_64-cpython-313/ManyWorldsInverseRadar.cpython-313-x86_64-linux-gnu.so scripts/ManyWorldsInverseRadar.so
