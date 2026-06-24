#pragma once
#include <chrono>

// Returns a monotonic clock time point
inline std::chrono::steady_clock::time_point stable_time() {
    return std::chrono::steady_clock::now();
}

// Number of ticks per second for the steady clock
constexpr long long stable_clocks_per_sec =
    std::chrono::steady_clock::period::den / std::chrono::steady_clock::period::num;

// Helper to convert duration to seconds as float
template<typename Rep, typename Period>
inline float to_seconds_float(const std::chrono::duration<Rep, Period>& d) {
    return std::chrono::duration<float>(d).count();
}

// Helper to convert duration to seconds as double
template<typename Rep, typename Period>
inline double to_seconds_double(const std::chrono::duration<Rep, Period>& d) {
    return std::chrono::duration<double>(d).count();
}