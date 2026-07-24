#include "stlq/common/timer.h"

#include <sstream>

namespace stlq {

Timer::Timer() : start_(std::chrono::steady_clock::now()) {}

void Timer::Reset() {
    start_ = std::chrono::steady_clock::now();
}

double Timer::ElapsedSeconds() const {
    using Duration = std::chrono::duration<double>;
    return std::chrono::duration_cast<Duration>(std::chrono::steady_clock::now() - start_).count();
}

std::string Timer::ReportSeconds(const std::string& label) const {
    std::ostringstream oss;
    oss << label << ": " << ElapsedSeconds() << "s";
    return oss.str();
}

}  // namespace stlq
