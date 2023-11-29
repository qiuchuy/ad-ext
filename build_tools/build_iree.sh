# This is borrowed from https://iree.dev/building-from-source/getting-started/
# Build IREE from source
cd /root/iree
cmake -G Ninja -B ../iree-build/ .
cmake --build ../iree-build/
cd /root
