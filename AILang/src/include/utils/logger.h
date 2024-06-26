#pragma once

#include <stdio.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ainl::core {

enum LogPriority {
  TracePriority,
  DebugPriority,
  InfoPriority,
  WarnPriority,
  ErrorPriority,
  CriticalPriority
};

class Logger {
private:
  LogPriority priority = DebugPriority;
  std::mutex logMutex;
  const char *filePath = 0;
  FILE *file = 0;

public:
  static void setPriority(LogPriority newPriority) {
    getInstance().priority = newPriority;
  }

  static void enableFileOutput() {
    Logger &loggerInstance = getInstance();
    loggerInstance.filePath = "/root/AILang/log";
    loggerInstance.enableFileOutput_();
  }

  static void enableFileOutput(const char *newFilePath) {
    Logger &loggerInstance = getInstance();
    loggerInstance.filePath = newFilePath;
    loggerInstance.enableFileOutput_();
  }

  template <typename... Args>
  static void Trace(const char *message, Args... args) {
    getInstance().log("[Trace]", TracePriority, message, args...);
  }

  template <typename... Args>
  static void Debug(const char *message, Args... args) {
    getInstance().log("[Debug] ", DebugPriority, message, args...);
  }

  template <typename... Args>
  static void Info(const char *message, Args... args) {
    getInstance().log("[Info] ", InfoPriority, message, args...);
  }

  template <typename... Args>
  static void Warn(const char *message, Args... args) {
    getInstance().log("[Warn] ", WarnPriority, message, args...);
  }

  template <typename... Args>
  static void Error(const char *message, Args... args) {
    getInstance().log("[Error] ", ErrorPriority, message, args...);
  }

  template <typename... Args>
  static void Critical(const char *message, Args... args) {
    getInstance().log("[Critical] ", CriticalPriority, message, args...);
  }

  template <typename... Args>
  static void Trace(int line, const char *sourceFile, const char *message,
                    Args... args) {
    getInstance().log(line, sourceFile, "[Trace] ", TracePriority, message,
                      args...);
  }

  template <typename... Args>
  static void Debug(int line, const char *sourceFile, const char *message,
                    Args... args) {
    getInstance().log(line, sourceFile, "[Debug] ", DebugPriority, message,
                      args...);
  }

  template <typename... Args>
  static void Info(int line, const char *sourceFile, const char *message,
                   Args... args) {
    getInstance().log(line, sourceFile, "[Info] ", InfoPriority, message,
                      args...);
  }

  template <typename... Args>
  static void Warn(int line, const char *sourceFile, const char *message,
                   Args... args) {
    getInstance().log(line, sourceFile, "[Warn] ", WarnPriority, message,
                      args...);
  }

  template <typename... Args>
  static void Error(int line, const char *sourceFile, const char *message,
                    Args... args) {
    getInstance().log(line, sourceFile, "[Error] ", ErrorPriority, message,
                      args...);
  }

  template <typename... Args>
  static void Critical(int line, const char *sourceFile, const char *message,
                       Args... args) {
    getInstance().log(line, sourceFile, "[Critical] ", CriticalPriority,
                      message, args...);
  }

private:
  Logger() {}

  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  ~Logger() { freeFile(); }

  static Logger &getInstance() {
    static Logger logger;
    return logger;
  }

  template <typename... Args>
  void log(const char *messagePriorityStr, LogPriority messagePriority,
           const char *message, Args... args) {
    if (priority <= messagePriority) {
      std::time_t current_time = std::time(0);
      std::tm *timestamp = std::localtime(&current_time);
      char buffer[80];
      strftime(buffer, 80, "%c", timestamp);
      std::scoped_lock lock(logMutex);

      char formattedMessage[1024];
      formatMessage(formattedMessage, sizeof(formattedMessage), message, args...);

      printf("%s %s%s\n", buffer, messagePriorityStr, formattedMessage);

      if (file) {
        fprintf(file, "%s %s%s\n", buffer, messagePriorityStr, formattedMessage);
      }
    }
  }

  template <typename... Args>
  void log(int line_number, const char *sourceFile,
           const char *messagePriorityStr, LogPriority messagePriority,
           const char *message, Args... args) {
    if (priority <= messagePriority) {
      std::time_t current_time = std::time(0);
      std::tm *timestamp = std::localtime(&current_time);
      char buffer[80];
      strftime(buffer, 80, "%c", timestamp);
      std::scoped_lock lock(logMutex);

      char formattedMessage[1024];
      formatMessage(formattedMessage, sizeof(formattedMessage), message, args...);

      printf("%s %s%s on line %d in %s\n", buffer, messagePriorityStr, formattedMessage, line_number, sourceFile);

      if (file) {
        fprintf(file, "%s %s%s on line %d in %s\n", buffer, messagePriorityStr, formattedMessage, line_number, sourceFile);
      }
    }
  }

  template <typename... Args>
  void formatMessage(char *buffer, size_t bufferSize, const char *format, Args... args) {
    snprintf(buffer, bufferSize, format, convertToCStr(args)...);
  }

  template <typename T>
  T convertToCStr(T value) {
    return value;
  }

  const char* convertToCStr(const std::string &value) {
    return value.c_str();
  }

  bool enableFileOutput_() {
    freeFile();

    file = std::fopen(filePath, "a");

    if (file == 0) {
      return false;
    }

    return true;
  }

  void freeFile() {
    if (file) {
      fclose(file);
      file = 0;
    }
  }
};

#define LOG_TRACE(Message, ...)                                                \
  (ainl::core::Logger::Trace(__LINE__, __FILE__, Message, __VA_ARGS__))
#define LOG_DEBUG(Message, ...)                                                \
  (ainl::core::Logger::Debug(__LINE__, __FILE__, Message, __VA_ARGS__))
#define LOG_INFO(Message, ...)                                                 \
  (ainl::core::Logger::Info(__LINE__, __FILE__, Message, __VA_ARGS__))
#define LOG_WARN(Message, ...)                                                 \
  (ainl::core::Logger::Warn(__LINE__, __FILE__, Message, __VA_ARGS__))
#define LOG_ERROR(Message, ...)                                                \
  (ainl::core::Logger::Error(__LINE__, __FILE__, Message, __VA_ARGS__))
#define LOG_CRITICAL(Message, ...)                                             \
  (ainl::core::Logger::Critical(__LINE__, __FILE__, Message, __VA_ARGS__))

class AINLError : public std::exception {
public:
  explicit AINLError(std::string message) : _message(std::move(message)) {}
  explicit AINLError(const char *message) : _message(message) {}
  const char *what() const noexcept override { return _message.c_str(); }

private:
  std::string _message;
};

} // namespace ainl::core
