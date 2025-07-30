mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j 1

cd ..
cp -r build/src/pybindmwir/PyBindMWIR.cpython-313-x86_64-linux-gnu.so scripts/PyMWIR/PyBindMWIR.so