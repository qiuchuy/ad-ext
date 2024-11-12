#pragma once

#include <string>

namespace ainl::core {
struct Device {
  enum class DeviceType : int {
    cpu = 0,
    gpu = 1,
  };
  DeviceType type;
  int id;
  Device() = default;
  constexpr Device(DeviceType type) : type(type), id(0) {}

  // Define member functions in a way that supports constexpr
  constexpr bool operator<(const Device &other) const {
    return type < other.type || (type == other.type && id < other.id);
  }
  constexpr bool operator==(const Device &other) const {
    return type == other.type && id == other.id;
  }
  constexpr bool operator!=(const Device &other) const {
    return !(*this == other);
  }

  size_t hash() const { return static_cast<size_t>(type); }

  std::string toString() const {
    switch (type) {
    case DeviceType::cpu:
      return "cpu";
    case DeviceType::gpu:
      return "gpu";
    }
    return "unknown";
  }
};

// Declare constexpr variables
static constexpr Device cpu = Device(Device::DeviceType::cpu);
static constexpr Device gpu = Device(Device::DeviceType::gpu);

const Device &getDefaultDevice();
} // namespace ainl::core
