#pragma once

#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace stlq {

enum class LogLevel {
    kInfo,
    kWarn,
    kError,
};

class Logger {
public:
    static Logger& Instance();
    void Log(LogLevel level, const std::string& message);
    void LogProgress(LogLevel level, const std::string& message, bool done);
    bool SetLogFile(const std::string& path, bool append, std::string* err);
    void ClearLogFile();

private:
    Logger() = default;
    std::mutex mutex_;
    bool progress_active_ = false;
    std::size_t progress_len_ = 0;
    std::string log_file_path_;
    std::unique_ptr<std::ofstream> log_file_;
    std::vector<std::string> history_;
};

void LogInfo(const std::string& message);
void LogWarn(const std::string& message);
void LogError(const std::string& message);
void LogInfoProgress(const std::string& message);
void LogInfoProgressDone(const std::string& message);
bool SetLogFile(const std::string& path, bool append, std::string* err);
void ClearLogFile();

}  // namespace stlq
