#include "logger.h"

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
