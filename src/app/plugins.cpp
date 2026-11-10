#include "plugins.h"

#include "i18n.h"
#include "platform.h"

#include "../core/model.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>

namespace app {

const std::string& Plugin::title(int lang) const {
    return lang >= 0 && lang < LANG_COUNT && !name[lang].empty() ? name[lang] : name[0];
}

const std::string& Plugin::description(int lang) const {
    return lang >= 0 && lang < LANG_COUNT && !desc[lang].empty() ? desc[lang] : desc[0];
}

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

// Parses the header comment of a plugin file; false if it is not a plugin.
static bool read_header(const std::filesystem::path& file, Plugin& p) {
    std::ifstream in(file);
    std::string line;
    bool marked = false;
    for (int n = 0; n < 40 && std::getline(in, line); ++n) {
        if (n == 0 && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
        line = trim(line);
        if (line.compare(0, 2, "#!") == 0 || line.empty()) continue;
        if (line[0] != '#') break;
        std::string body = trim(line.substr(1));
        if (!marked) {
            if (body != "kor-plugin") return false;
            marked = true;
            continue;
        }
        size_t colon = body.find(':');
        if (colon == std::string::npos) continue;
        std::string key = trim(body.substr(0, colon)), value = trim(body.substr(colon + 1));
        static const char* const suffix[LANG_COUNT] = {"", ".tr", ".ru"};
        for (int l = 0; l < LANG_COUNT; ++l) {
            if (key == std::string("name") + suffix[l]) p.name[l] = value;
            if (key == std::string("description") + suffix[l]) p.desc[l] = value;
        }
        if (key == "args") p.args = value;
    }
    if (!marked) return false;
    if (p.name[0].empty()) p.name[0] = file.stem().u8string();
    return true;
}

std::vector<Plugin> scan_plugins(const std::vector<std::string>& dirs) {
    namespace fs = std::filesystem;
    std::vector<Plugin> out;
    std::set<std::string> seen;
    for (auto& d : dirs) {
        std::vector<fs::path> files;
        std::error_code ec;
        for (fs::directory_iterator it(kor::upath(d), ec), end; !ec && it != end; it.increment(ec))
            if (it->path().extension() == ".py") files.push_back(it->path());
        std::sort(files.begin(), files.end());
        for (auto& f : files) {
            Plugin p;
            p.path = f.u8string();
            p.file = f.filename().u8string();
            if (seen.count(p.file) || !read_header(f, p)) continue;
            seen.insert(p.file);
            out.push_back(p);
        }
    }
    return out;
}

std::vector<std::string> split_args(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    bool quoted = false, any = false;
    for (char c : s) {
        if (c == '"') { quoted = !quoted; any = true; }
        else if ((c == ' ' || c == '\t') && !quoted) {
            if (any) out.push_back(cur);
            cur.clear();
            any = false;
        } else {
            cur += c;
            any = true;
        }
    }
    if (any) out.push_back(cur);
    return out;
}

// ---------------------------------------------------------------- Python

PythonFinder::~PythonFinder() {
    if (th_.joinable()) th_.join();
}

void PythonFinder::start() {
    if (searching_) return;
    if (th_.joinable()) th_.join();
    searching_ = true;
    th_ = std::thread([this] {
        std::string v;
        std::vector<std::string> cmd = find_python(v);
        {
            std::lock_guard<std::mutex> lk(mu_);
            cmd_ = cmd;
            version_ = v;
        }
        searching_ = false;
    });
}

bool PythonFinder::found() {
    std::lock_guard<std::mutex> lk(mu_);
    return !cmd_.empty();
}

std::string PythonFinder::version() {
    std::lock_guard<std::mutex> lk(mu_);
    return version_;
}

std::vector<std::string> PythonFinder::command() {
    std::lock_guard<std::mutex> lk(mu_);
    return cmd_;
}

// ---------------------------------------------------------------- runner

PluginRunner::~PluginRunner() {
    cancel_ = true;
    if (th_.joinable()) th_.join();
}

bool PluginRunner::start(const std::vector<std::string>& cmd, const std::vector<std::pair<std::string, std::string>>& env) {
    if (running_ || cmd.empty()) return false;
    if (th_.joinable()) th_.join();
    {
        std::lock_guard<std::mutex> lk(mu_);
        out_.clear();
        finished_ = false;
    }
    cancel_ = false;
    running_ = true;
    th_ = std::thread([this, cmd, env] {
        for (auto& kv : env) set_env(kv.first.c_str(), kv.second.c_str());
        int code = run_process(cmd, [this](const std::string& s) {
            std::lock_guard<std::mutex> lk(mu_);
            out_ += s;
            const size_t cap = 256 * 1024;  // keep the tail of very chatty plugins
            if (out_.size() > cap) out_.erase(0, out_.size() - cap);
        }, &cancel_);
        {
            std::lock_guard<std::mutex> lk(mu_);
            finished_ = true;
            code_ = code;
        }
        running_ = false;
    });
    return true;
}

std::string PluginRunner::output() {
    std::lock_guard<std::mutex> lk(mu_);
    return out_;
}

void PluginRunner::clear_output() {
    std::lock_guard<std::mutex> lk(mu_);
    out_.clear();
}

bool PluginRunner::take_finished(int& code) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!finished_) return false;
    finished_ = false;
    code = code_;
    return true;
}

}  // namespace app
