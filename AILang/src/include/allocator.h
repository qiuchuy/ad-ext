#pragma once

#include <functional>

#include "utils/logger.h"

namespace ainl::core::allocator {
class Buffer {
private:
  void *ptr_;

public:
  Buffer(void *ptr) : ptr_(ptr){};
  void *ptr() { return ptr_; }
  void *ptr() const { return ptr_; }
};

Buffer malloc(size_t size);

void free(Buffer buffer);

class Allocator {
public:
  virtual Buffer malloc(size_t size) = 0;
  virtual void free(Buffer buffer) = 0;

  Allocator() = default;
  Allocator(const Allocator &other) = delete;
  Allocator(Allocator &&other) = delete;
  Allocator &operator=(const Allocator &other) = delete;
  Allocator &operator=(Allocator &&other) = delete;
  virtual ~Allocator() = default;
};

Allocator &allocator();

class CPUAllocator : public Allocator {
public:
  Buffer malloc(size_t size) override;
  void free(Buffer buffer) override;

private:
  CPUAllocator();
  std::function<Buffer(size_t)> allocateStrategy_;
  std::function<void(Buffer)> freeStrategy_;
  friend Allocator &allocator();
};

}; // namespace ainl::core::allocator
