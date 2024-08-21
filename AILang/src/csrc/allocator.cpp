#include "allocator.h"

#include <cstdlib>

#include "utils/logger.h"

namespace ainl::core::allocator {
Buffer malloc(size_t size) {
  auto buffer = allocator().malloc(size);
  if (size && !buffer.ptr()) {
    throw AINLError("[allocator] Failed to allocate memory.");
  }
  return buffer;
}

void free(Buffer buffer) { return allocator().free(buffer); }

CPUAllocator::CPUAllocator() {
  allocateStrategy_ = [](size_t size) {
    return Buffer(static_cast<void *>(std::malloc(size)));
  };
  freeStrategy_ = [](Buffer buffer) { std::free(buffer.ptr()); };
}

Buffer CPUAllocator::malloc(size_t size) {
  if (size) {
    return allocateStrategy_(size);
  } else {
    throw AINLError("[allocator] Try to allocate 0 bytes memory.");
  }
}

void CPUAllocator::free(Buffer buffer) {
  if (buffer.ptr()) {
    freeStrategy_(buffer);
  } else {
    throw AINLError("[allocator] Try to free nullptr.");
  }
}

Allocator &allocator() {
  static CPUAllocator allocator;
  return allocator;
}

} // namespace ainl::core::allocator
