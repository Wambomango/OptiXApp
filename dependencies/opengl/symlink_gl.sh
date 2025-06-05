SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# sudo apt install libglvnd-dev
ln -s /usr/include/EGL "$SCRIPT_DIR/include/"
ln -s /usr/include/KHR "$SCRIPT_DIR/include/"
ln -s /usr/include/GL "$SCRIPT_DIR/include/"
ln -s /usr/lib/x86_64-linux-gnu/libOpenGL.so "$SCRIPT_DIR/libOpenGL.so"
ln -s /usr/lib/x86_64-linux-gnu/libEGL.so "$SCRIPT_DIR/libEGL.so"