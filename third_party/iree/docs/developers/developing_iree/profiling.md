# Profiling

IREE [benchmarking](./benchmarking.md) gives us an accurate and reproducible
view of program performance at specific levels of granularity. To analyze system
behavior in more depth, there are various ways to
[profile](https://en.wikipedia.org/wiki/Profiling_(computer_programming)) IREE.

## Tracy

Tracy is a profiler that's been used for a wide range of profiling tasks on
IREE. Refer to [profiling_with_tracy.md](./profiling_with_tracy.md).

## Vulkan GPU Profiling

[Tracy](./profiling_with_tracy.md) offers great insights into CPU/GPU
interactions and Vulkan API usage details. However, information at a finer
granularity, especially inside a particular shader dispatch, is missing. To
supplement general purpose tools like Tracy, vendor-specific tools can be used.
Refer to [profiling_vulkan_gpu.md](./profiling_vulkan_gpu.md).

## CPU cache and other CPU event profiling

For some advanced CPU profiling needs such as querying CPU cache and other
events, one may need to use some OS-specific profilers. See
[profiling_cpu_events.md](./profiling_cpu_events.md).
