#pragma once

#include "levo.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace levo_cli {

enum class progress_mode { plain, json, none };

inline progress_mode parse_progress_mode(const std::string & value) {
    if (value == "plain") return progress_mode::plain;
    if (value == "json") return progress_mode::json;
    if (value == "none") return progress_mode::none;
    throw std::invalid_argument("--progress must be plain, json, or none");
}

inline double parse_progress_interval(const std::string & value) {
    std::size_t used = 0;
    try {
        const double parsed = std::stod(value, &used);
        if (used != value.size() || !std::isfinite(parsed) || parsed < 0.0) {
            throw std::invalid_argument("interval");
        }
        return parsed;
    } catch (const std::exception &) {
        throw std::invalid_argument("--progress-interval must be a finite non-negative number");
    }
}

inline std::atomic<bool> interrupted{false};

inline void signal_handler(int) { interrupted.store(true, std::memory_order_relaxed); }

inline void install_signal_handlers() {
    interrupted.store(false, std::memory_order_relaxed);
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
}

inline bool cancellation_requested() { return interrupted.load(std::memory_order_relaxed); }

inline const char * stage_name(levo::generation_stage stage) {
    switch (stage) {
        case levo::generation_stage::initializing_backend: return "initializing_backend";
        case levo::generation_stage::loading_model: return "loading_model";
        case levo::generation_stage::preparing_conditioning: return "preparing_conditioning";
        case levo::generation_stage::prefilling: return "prefilling";
        case levo::generation_stage::generating: return "generating";
        case levo::generation_stage::complete: return "complete";
    }
    return "unknown";
}

inline const char * stage_name(levo::render_stage stage) {
    switch (stage) {
        case levo::render_stage::loading_tokens: return "loading_tokens";
        case levo::render_stage::initializing_backend: return "initializing_backend";
        case levo::render_stage::loading_flow: return "loading_flow";
        case levo::render_stage::generating_latents: return "generating_latents";
        case levo::render_stage::releasing_flow: return "releasing_flow";
        case levo::render_stage::loading_vae: return "loading_vae";
        case levo::render_stage::decoding_window: return "decoding_window";
        case levo::render_stage::assembling_audio: return "assembling_audio";
        case levo::render_stage::complete: return "complete";
    }
    return "unknown";
}

inline std::string duration_text(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) return "--:--";
    const auto rounded = static_cast<unsigned long long>(seconds + 0.5);
    const auto hours = rounded / 3600U;
    const auto minutes = (rounded % 3600U) / 60U;
    const auto remainder = rounded % 60U;
    std::ostringstream out;
    if (hours != 0) out << hours << ':' << std::setw(2) << std::setfill('0') << minutes << ':';
    else out << minutes << ':';
    out << std::setw(2) << std::setfill('0') << remainder;
    return out.str();
}

struct rate_eta {
    double rate = 0.0;
    double eta = -1.0;
};

inline rate_eta generation_rate(const levo::generation_progress & value) {
    if (value.stage != levo::generation_stage::generating || value.completed_steps == 0 ||
        value.stage_elapsed_seconds <= 0.0 || value.total_steps < value.completed_steps) return {};
    const double rate = static_cast<double>(value.completed_steps) / value.stage_elapsed_seconds;
    return {rate, rate > 0.0 ? static_cast<double>(value.total_steps - value.completed_steps) / rate : -1.0};
}

inline rate_eta render_rate(const levo::render_progress & value) {
    if (value.stage == levo::render_stage::generating_latents && value.current_window != 0 &&
        value.total_steps != 0 && value.stage_elapsed_seconds > 0.0) {
        const std::size_t completed = (value.current_window - 1U) * value.total_steps + value.completed_steps;
        const std::size_t total = value.total_windows * value.total_steps;
        if (completed == 0 || completed > total) return {};
        const double rate = static_cast<double>(completed) / value.stage_elapsed_seconds;
        return {rate, rate > 0.0 ? static_cast<double>(total - completed) / rate : -1.0};
    }
    if (value.stage == levo::render_stage::decoding_window && value.completed_windows != 0 &&
        value.stage_elapsed_seconds > 0.0 && value.completed_windows <= value.total_windows) {
        const double rate = static_cast<double>(value.completed_windows) / value.stage_elapsed_seconds;
        return {rate, rate > 0.0 ? static_cast<double>(value.total_windows - value.completed_windows) / rate : -1.0};
    }
    return {};
}

inline std::string generation_json(const levo::generation_progress & value) {
    const rate_eta measured = generation_rate(value);
    std::ostringstream out;
    out << std::setprecision(9)
        << "{\"type\":\"generation_progress\",\"stage\":\"" << stage_name(value.stage)
        << "\",\"completed_steps\":" << value.completed_steps
        << ",\"total_steps\":" << value.total_steps
        << ",\"requested_frames\":" << value.requested_frames
        << ",\"ended\":[" << (value.ended[0] ? "true" : "false") << ','
        << (value.ended[1] ? "true" : "false") << ',' << (value.ended[2] ? "true" : "false") << ']'
        << ",\"elapsed_seconds\":" << value.elapsed_seconds
        << ",\"stage_elapsed_seconds\":" << value.stage_elapsed_seconds
        << ",\"steps_per_second\":";
    if (measured.rate > 0.0) out << measured.rate; else out << "null";
    out << ",\"eta_seconds\":";
    if (measured.eta >= 0.0) out << measured.eta; else out << "null";
    out << '}';
    return out.str();
}

inline std::string render_json(const levo::render_progress & value) {
    const rate_eta measured = render_rate(value);
    std::ostringstream out;
    out << std::setprecision(9)
        << "{\"type\":\"render_progress\",\"stage\":\"" << stage_name(value.stage)
        << "\",\"completed_windows\":" << value.completed_windows
        << ",\"total_windows\":" << value.total_windows
        << ",\"current_window\":" << value.current_window
        << ",\"completed_steps\":" << value.completed_steps
        << ",\"total_steps\":" << value.total_steps
        << ",\"elapsed_seconds\":" << value.elapsed_seconds
        << ",\"stage_elapsed_seconds\":" << value.stage_elapsed_seconds
        << ",\"units_per_second\":";
    if (measured.rate > 0.0) out << measured.rate; else out << "null";
    out << ",\"eta_seconds\":";
    if (measured.eta >= 0.0) out << measured.eta; else out << "null";
    out << '}';
    return out.str();
}

inline std::string generation_plain(const levo::generation_progress & value) {
    const rate_eta measured = generation_rate(value);
    std::ostringstream out;
    out << '[' << duration_text(value.elapsed_seconds) << "] " << stage_name(value.stage);
    if (value.stage == levo::generation_stage::generating && value.total_steps != 0) {
        const double percent = 100.0 * static_cast<double>(value.completed_steps) / value.total_steps;
        out << ' ' << value.completed_steps << '/' << value.total_steps << " steps ("
            << std::fixed << std::setprecision(1) << percent << "%)";
        if (measured.rate > 0.0) out << ", " << std::setprecision(2) << measured.rate << " step/s";
        if (measured.eta >= 0.0) out << ", ETA " << duration_text(measured.eta);
    }
    return out.str();
}

inline std::string render_plain(const levo::render_progress & value) {
    const rate_eta measured = render_rate(value);
    std::ostringstream out;
    out << '[' << duration_text(value.elapsed_seconds) << "] " << stage_name(value.stage);
    if (value.stage == levo::render_stage::generating_latents && value.current_window != 0) {
        out << " window " << value.current_window << '/' << value.total_windows
            << ", Euler " << value.completed_steps << '/' << value.total_steps;
        if (measured.rate > 0.0) out << ", " << std::fixed << std::setprecision(2) << measured.rate << " step/s";
        if (measured.eta >= 0.0) out << ", ETA " << duration_text(measured.eta);
    } else if (value.total_windows != 0) {
        out << ' ' << value.completed_windows << '/' << value.total_windows << " windows";
    }
    return out.str();
}

template <typename Event>
class progress_writer final {
public:
    using stage_type = decltype(Event{}.stage);

    progress_writer(progress_mode mode, double interval_seconds, std::ostream & output = std::cerr)
        : mode_(mode), interval_(interval_seconds), output_(output) {
        if (mode_ != progress_mode::none && interval_ > 0.0) {
            worker_ = std::thread([this] { heartbeat(); });
        }
    }

    progress_writer(const progress_writer &) = delete;
    progress_writer & operator=(const progress_writer &) = delete;

    ~progress_writer() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    void update(const Event & event) {
        if (mode_ == progress_mode::none) return;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now = std::chrono::steady_clock::now();
        const bool stage_changed = !has_event_ || event.stage != latest_.stage;
        latest_ = event;
        received_ = now;
        has_event_ = true;
        complete_ = is_complete(event);
        if (stage_changed || complete_ || interval_ == 0.0 ||
            std::chrono::duration<double>(now - emitted_).count() >= interval_) {
            emit_locked(event, now);
        }
        condition_.notify_all();
    }

private:
    static bool is_complete(const levo::generation_progress & event) {
        return event.stage == levo::generation_stage::complete;
    }
    static bool is_complete(const levo::render_progress & event) {
        return event.stage == levo::render_stage::complete;
    }
    static std::string plain(const levo::generation_progress & event) { return generation_plain(event); }
    static std::string plain(const levo::render_progress & event) { return render_plain(event); }
    static std::string json(const levo::generation_progress & event) { return generation_json(event); }
    static std::string json(const levo::render_progress & event) { return render_json(event); }

    void emit_locked(const Event & event, std::chrono::steady_clock::time_point now) {
        output_ << (mode_ == progress_mode::json ? json(event) : plain(event)) << '\n';
        output_.flush();
        emitted_ = now;
    }

    void heartbeat() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stopping_) {
            condition_.wait_for(lock, std::chrono::duration<double>(interval_));
            if (stopping_) break;
            if (!has_event_ || complete_) continue;
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - emitted_).count() < interval_) continue;
            Event heartbeat_event = latest_;
            const double added = std::chrono::duration<double>(now - received_).count();
            heartbeat_event.elapsed_seconds += added;
            heartbeat_event.stage_elapsed_seconds += added;
            emit_locked(heartbeat_event, now);
        }
    }

    progress_mode mode_;
    double interval_;
    std::ostream & output_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::thread worker_;
    Event latest_{};
    bool has_event_ = false;
    bool complete_ = false;
    bool stopping_ = false;
    std::chrono::steady_clock::time_point received_ = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point emitted_{};
};

using generation_progress_writer = progress_writer<levo::generation_progress>;
using render_progress_writer = progress_writer<levo::render_progress>;

} // namespace levo_cli
