#include "utils/logger.h"

namespace ainl::core {
std::string logLevelToString(LogLevel level) {
  static std::unordered_map<LogLevel, std::string> levelStrings = {
      {LogLevel::DEBUG, "DEBUG"},
      {LogLevel::INFO, "INFO"},
      {LogLevel::WARNING, "WARNING"},
      {LogLevel::ERROR, "ERROR"}};

  auto it = levelStrings.find(level);
  if (it != levelStrings.end()) {
    return it->second;
  } else {
    return "UNKNOWN";
  }
}
} // namespace ainl::core