// Kor Studio - Python plugins: discovery, Python detection and a background runner.
//
// A plugin is a .py file whose first comment lines start with "# kor-plugin",
// followed by "# key: value" lines (name, description, args; name.tr etc. for
// translations). See docs/plugins.md.
#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace app {

struct Plugin {
    std::string path, file;     // UTF-8
    std::string name[3], desc[3];  // per Lang value; English is the fallback
    std::string args;           // default arguments
    const std::string& title(int lang) const;
    const std::string& description(int lang) const;
};

// Reads every plugin in `dirs` (sorted by file name, first folder wins on duplicates).
std::vector<Plugin> scan_plugins(const std::vector<std::string>& dirs);

// Splits a command line into arguments; double quotes group words.
std::vector<std::string> split_args(const std::string& s);

// Looks for a Python 3 interpreter on a background thread.
class PythonFinder {
public:
    ~PythonFinder();
    void start();
    bool searching() const { return searching_; }
    bool found();
    std::string version();
    std::vector<std::string> command();

private:
    std::thread th_;
    std::atomic<bool> searching_{false};
    std::mutex mu_;
    std::vector<std::string> cmd_;
    std::string version_;
};

// Runs one program at a time on a background thread and collects its output.
class PluginRunner {
public:
    ~PluginRunner();
    // `env` is applied to this process before the program starts (children inherit it).
    bool start(const std::vector<std::string>& cmd, const std::vector<std::pair<std::string, std::string>>& env);
    void cancel() { cancel_ = true; }
    bool running() const { return running_; }
    std::string output();
    void clear_output();
    // True once after each finished run; `code` gets the exit code (-1 not started, -2 cancelled).
    bool take_finished(int& code);

private:
    std::thread th_;
    std::atomic<bool> running_{false}, cancel_{false};
    std::mutex mu_;
    std::string out_;
    bool finished_ = false;
    int code_ = 0;
};

}  // namespace app
