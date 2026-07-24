#include "stlq/common/logger.h"

#include <fstream>
#include <iostream>

namespace stlq {

namespace {

std::string MakeTaggedLine(const char* tag, const std::string& message) {
    return std::string("[") + tag + "] " + message;
}

}  // namespace

Logger& Logger::Instance() {
    static Logger instance;
    return instance;
}

void Logger::Log(LogLevel level, const std::string& message) {
    const char* tag = "INFO";
    if (level == LogLevel::kWarn) {
        tag = "WARN";
    } else if (level == LogLevel::kError) {
        tag = "ERROR";
    }

    std::lock_guard<std::mutex> guard(mutex_);
    if (progress_active_) {
        std::cerr << "\n";
        progress_active_ = false;
        progress_len_ = 0;
    }
    const std::string line = MakeTaggedLine(tag, message);
    std::cerr << line << "\n";
    if (log_file_ && log_file_->is_open()) {
        (*log_file_) << line << "\n";
        log_file_->flush();
    } else {
        history_.push_back(line);
    }
}

void Logger::LogProgress(LogLevel level, const std::string& message, bool done) {
    const char* tag = "INFO";
    if (level == LogLevel::kWarn) {
        tag = "WARN";
    } else if (level == LogLevel::kError) {
        tag = "ERROR";
    }

    std::lock_guard<std::mutex> guard(mutex_);
    std::cerr << "\r[" << tag << "] " << message;
    const std::size_t new_len = message.size();
    if (progress_active_ && new_len < progress_len_) {
        std::cerr << std::string(progress_len_ - new_len, ' ');
    }
    progress_active_ = !done;
    progress_len_ = done ? 0 : new_len;
    if (done) {
        std::cerr << "\n";
    }
    std::cerr.flush();
    if (log_file_ && log_file_->is_open() && done) {
        (*log_file_) << MakeTaggedLine(tag, message) << "\n";
        log_file_->flush();
    } else if (done) {
        history_.push_back(MakeTaggedLine(tag, message));
    }
}

bool Logger::SetLogFile(const std::string& path, bool append, std::string* err) {
    std::lock_guard<std::mutex> guard(mutex_);
    log_file_.reset();
    log_file_path_.clear();
    if (path.empty()) {
        return true;
    }
    auto out = std::make_unique<std::ofstream>(path,
                                               std::ios::out | std::ios::binary |
                                                   (append ? std::ios::app : std::ios::trunc));
    if (!out->is_open()) {
        if (err) *err = "Failed to open log file: " + path;
        return false;
    }
    for (const std::string& line : history_) {
        (*out) << line << "\n";
    }
    out->flush();
    history_.clear();
    log_file_path_ = path;
    log_file_ = std::move(out);
    return true;
}

void Logger::ClearLogFile() {
    std::lock_guard<std::mutex> guard(mutex_);
    log_file_.reset();
    log_file_path_.clear();
}

void LogInfo(const std::string& message) {
    Logger::Instance().Log(LogLevel::kInfo, message);
}

void LogWarn(const std::string& message) {
    Logger::Instance().Log(LogLevel::kWarn, message);
}

void LogError(const std::string& message) {
    Logger::Instance().Log(LogLevel::kError, message);
}

void LogInfoProgress(const std::string& message) {
    Logger::Instance().LogProgress(LogLevel::kInfo, message, /*done=*/false);
}

void LogInfoProgressDone(const std::string& message) {
    Logger::Instance().LogProgress(LogLevel::kInfo, message, /*done=*/true);
}

bool SetLogFile(const std::string& path, bool append, std::string* err) {
    return Logger::Instance().SetLogFile(path, append, err);
}

void ClearLogFile() {
    Logger::Instance().ClearLogFile();
}

}  // namespace stlq
