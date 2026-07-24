#pragma once

#include <chrono>
#include <string>

namespace stlq {

class Timer {
public:
    Timer();
    void Reset();
    double ElapsedSeconds() const;
    std::string ReportSeconds(const std::string& label) const;

private:
    std::chrono::steady_clock::time_point start_;
};

}  // namespace stlq
