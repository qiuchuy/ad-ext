#include "ailang/Core/Device.h"

namespace ainl::core {

const Device &getDefaultDevice() {
  static Device device(Device::DeviceType::cpu);
  return device;
}

} // namespace ainl::core