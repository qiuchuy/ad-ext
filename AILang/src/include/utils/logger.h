#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

namespace ainl::core {

enum class LogLevel { DEBUG, INFO, WARNING, ERROR };
constexpr LogLevel currentLogLevel = LogLevel::DEBUG;

std::string logLevelToString(LogLevel level);

#define LOG(level, message)                                                    \
  do {                                                                         \
    if (static_cast<int>(level) >= static_cast<int>(currentLogLevel)) {        \
      std::cout << "[" << logLevelToString(level) << "] " << __FILE__ << ":"   \
                << __LINE__ << " - " << message << std::endl;                  \
    }                                                                          \
  } while (0)

#define DEBUG(message) LOG(LogLevel::DEBUG, message);
#define INFO(message) LOG(LogLevel::INFO, message);
#define WARNING(message) LOG(LogLevel::WARNING, message);
#define ERROR(message) LOG(LogLevel::ERROR, message);

class AINLError : public std::exception {
public:
  explicit AINLError(std::string message) : _message(message) {}
  explicit AINLError(const char *message) : _message(message) {}
  const char *what() const noexcept override { return _message.c_str(); }

private:
  std::string _message;
};
} // namespace ainl::core