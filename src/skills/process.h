// Kor Studio - a child process we talk to line by line (standard input and
// output; standard error is merged into the output). Strings are UTF-8.
#pragma once
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace kor {

using EnvList = std::vector<std::pair<std::string, std::string>>;

class Process {
public:
    Process();
    ~Process();
    Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;

    // `env` is added to (or replaces entries of) this program's environment.
    bool start(const std::vector<std::string>& cmd, const EnvList& env, std::string* err);
    bool write_line(const std::string& line);
    void close_input();
    // 1 = `line` was read (without the line break), 0 = timed out, -1 = the
    // output is closed and drained.
    int read_line(std::string& line, int timeout_ms);
    // Exit code once the process has ended within `timeout_ms`, otherwise -1000.
    int wait(int timeout_ms);
    void kill();

private:
    void reader();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::thread th_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::string> lines_;
    std::string partial_;
    bool eof_ = false;
};

// Runs a program to completion and returns its exit code (-1 if it could not
// start); all output goes to `out`.
int run_capture(const std::vector<std::string>& cmd, const EnvList& env, std::string& out, int timeout_ms);

}  // namespace kor
