#include "process.h"

#include <algorithm>
#include <chrono>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace kor {

#if defined(_WIN32)

static std::wstring widen(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (n <= 1) return L"";
    std::wstring w((size_t)n - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    return w;
}

// Quotes one argument so that CommandLineToArgvW (and the C runtime) read it back unchanged.
static std::wstring quote_arg(const std::wstring& a) {
    std::wstring q = L"\"";
    size_t slashes = 0;
    for (wchar_t c : a) {
        if (c == L'\\') { ++slashes; continue; }
        if (c == L'"') q.append(slashes * 2 + 1, L'\\');
        else q.append(slashes, L'\\');
        slashes = 0;
        q += c;
    }
    q.append(slashes * 2, L'\\');
    return q + L"\"";
}

// The current environment with `env` applied, as a CreateProcessW block.
static std::vector<wchar_t> env_block(const EnvList& env) {
    std::vector<std::wstring> vars;
    if (wchar_t* cur = GetEnvironmentStringsW()) {
        for (wchar_t* p = cur; *p; p += wcslen(p) + 1) vars.push_back(p);
        FreeEnvironmentStringsW(cur);
    }
    for (auto& kv : env) {
        std::wstring k = widen(kv.first), entry = k + L"=" + widen(kv.second);
        bool replaced = false;
        for (auto& v : vars)
            if (v.size() > k.size() && v[k.size()] == L'=' && CompareStringOrdinal(v.c_str(), (int)k.size(), k.c_str(), (int)k.size(), TRUE) == CSTR_EQUAL) {
                v = entry;
                replaced = true;
            }
        if (!replaced) vars.push_back(entry);
    }
    std::vector<wchar_t> block;
    for (auto& v : vars) {
        block.insert(block.end(), v.begin(), v.end());
        block.push_back(L'\0');
    }
    block.push_back(L'\0');
    return block;
}

struct Process::Impl {
    HANDLE proc = nullptr, in_w = nullptr, out_r = nullptr;
};

Process::~Process() {
    kill();
    if (th_.joinable()) th_.join();
    if (impl_) {
        if (impl_->in_w) CloseHandle(impl_->in_w);
        if (impl_->out_r) CloseHandle(impl_->out_r);
        if (impl_->proc) CloseHandle(impl_->proc);
    }
}

bool Process::start(const std::vector<std::string>& cmd, const EnvList& env, std::string* err) {
    if (cmd.empty()) return false;
    std::wstring line;
    for (auto& a : cmd) {
        if (!line.empty()) line += L' ';
        line += quote_arg(widen(a));
    }
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE in_r = nullptr, in_w = nullptr, out_r = nullptr, out_w = nullptr;
    if (!CreatePipe(&in_r, &in_w, &sa, 0)) return false;
    if (!CreatePipe(&out_r, &out_w, &sa, 0)) {
        CloseHandle(in_r);
        CloseHandle(in_w);
        return false;
    }
    SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in_r;
    si.hStdOutput = out_w;
    si.hStdError = out_w;
    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));
    std::vector<wchar_t> buf(line.begin(), line.end());
    buf.push_back(L'\0');
    std::vector<wchar_t> block = env_block(env);
    BOOL ok = CreateProcessW(nullptr, buf.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,
                             block.data(), nullptr, &si, &pi);
    CloseHandle(in_r);
    CloseHandle(out_w);
    if (!ok) {
        if (err) *err = "could not start " + cmd[0] + " (error " + std::to_string(GetLastError()) + ")";
        CloseHandle(in_w);
        CloseHandle(out_r);
        return false;
    }
    CloseHandle(pi.hThread);
    impl_ = std::make_unique<Impl>();
    impl_->proc = pi.hProcess;
    impl_->in_w = in_w;
    impl_->out_r = out_r;
    th_ = std::thread([this] { reader(); });
    return true;
}

bool Process::write_line(const std::string& line) {
    if (!impl_ || !impl_->in_w) return false;
    std::string s = line + "\n";
    const char* p = s.data();
    size_t left = s.size();
    while (left > 0) {
        DWORD n = 0;
        if (!WriteFile(impl_->in_w, p, (DWORD)std::min<size_t>(left, 1 << 20), &n, nullptr) || n == 0) return false;
        p += n;
        left -= n;
    }
    return true;
}

void Process::close_input() {
    if (impl_ && impl_->in_w) {
        CloseHandle(impl_->in_w);
        impl_->in_w = nullptr;
    }
}

void Process::reader() {
    char buf[8192];
    for (;;) {
        DWORD n = 0;
        if (!ReadFile(impl_->out_r, buf, sizeof buf, &n, nullptr) || n == 0) break;
        std::lock_guard<std::mutex> lk(mu_);
        partial_.append(buf, n);
        size_t pos;
        while ((pos = partial_.find('\n')) != std::string::npos) {
            std::string l = partial_.substr(0, pos);
            if (!l.empty() && l.back() == '\r') l.pop_back();
            lines_.push_back(std::move(l));
            partial_.erase(0, pos + 1);
        }
        cv_.notify_all();
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (!partial_.empty()) lines_.push_back(partial_);
    partial_.clear();
    eof_ = true;
    cv_.notify_all();
}

int Process::wait(int timeout_ms) {
    if (!impl_ || !impl_->proc) return -1;
    if (WaitForSingleObject(impl_->proc, (DWORD)std::max(0, timeout_ms)) != WAIT_OBJECT_0) return -1000;
    DWORD code = 0;
    GetExitCodeProcess(impl_->proc, &code);
    return (int)code;
}

void Process::kill() {
    if (impl_ && impl_->proc && WaitForSingleObject(impl_->proc, 0) != WAIT_OBJECT_0) TerminateProcess(impl_->proc, 1);
    close_input();
}

#else  // POSIX

struct Process::Impl {
    pid_t pid = -1;
    int in_w = -1, out_r = -1;
    int code = -1000;
};

Process::~Process() {
    kill();
    if (th_.joinable()) th_.join();
    if (impl_) {
        if (impl_->in_w >= 0) close(impl_->in_w);
        if (impl_->out_r >= 0) close(impl_->out_r);
        wait(0);
    }
}

bool Process::start(const std::vector<std::string>& cmd, const EnvList& env, std::string* err) {
    if (cmd.empty()) return false;
    int in_p[2], out_p[2];
    if (pipe(in_p) != 0) return false;
    if (pipe(out_p) != 0) {
        close(in_p[0]);
        close(in_p[1]);
        return false;
    }
    pid_t pid = fork();
    if (pid < 0) {
        close(in_p[0]); close(in_p[1]); close(out_p[0]); close(out_p[1]);
        if (err) *err = "fork failed";
        return false;
    }
    if (pid == 0) {
        dup2(in_p[0], 0);
        dup2(out_p[1], 1);
        dup2(out_p[1], 2);
        close(in_p[0]); close(in_p[1]); close(out_p[0]); close(out_p[1]);
        for (auto& kv : env) setenv(kv.first.c_str(), kv.second.c_str(), 1);
        std::vector<char*> argv;
        for (auto& a : cmd) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        execvp(argv[0], argv.data());
        _exit(127);
    }
    close(in_p[0]);
    close(out_p[1]);
    signal(SIGPIPE, SIG_IGN);
    impl_ = std::make_unique<Impl>();
    impl_->pid = pid;
    impl_->in_w = in_p[1];
    impl_->out_r = out_p[0];
    th_ = std::thread([this] { reader(); });
    return true;
}

bool Process::write_line(const std::string& line) {
    if (!impl_ || impl_->in_w < 0) return false;
    std::string s = line + "\n";
    const char* p = s.data();
    size_t left = s.size();
    while (left > 0) {
        ssize_t n = ::write(impl_->in_w, p, left);
        if (n <= 0) return false;
        p += n;
        left -= (size_t)n;
    }
    return true;
}

void Process::close_input() {
    if (impl_ && impl_->in_w >= 0) {
        close(impl_->in_w);
        impl_->in_w = -1;
    }
}

void Process::reader() {
    char buf[8192];
    for (;;) {
        ssize_t n = ::read(impl_->out_r, buf, sizeof buf);
        if (n <= 0) break;
        std::lock_guard<std::mutex> lk(mu_);
        partial_.append(buf, (size_t)n);
        size_t pos;
        while ((pos = partial_.find('\n')) != std::string::npos) {
            lines_.push_back(partial_.substr(0, pos));
            partial_.erase(0, pos + 1);
        }
        cv_.notify_all();
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (!partial_.empty()) lines_.push_back(partial_);
    partial_.clear();
    eof_ = true;
    cv_.notify_all();
}

int Process::wait(int timeout_ms) {
    if (!impl_ || impl_->pid < 0) return -1;
    if (impl_->code != -1000) return impl_->code;
    auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max(0, timeout_ms));
    for (;;) {
        int st = 0;
        pid_t r = waitpid(impl_->pid, &st, WNOHANG);
        if (r == impl_->pid) {
            impl_->code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
            return impl_->code;
        }
        if (r < 0 || std::chrono::steady_clock::now() >= until) return r < 0 ? -1 : -1000;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void Process::kill() {
    if (impl_ && impl_->pid > 0 && wait(0) == -1000) ::kill(impl_->pid, SIGKILL);
    close_input();
}

#endif

Process::Process() = default;

int Process::read_line(std::string& line, int timeout_ms) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait_for(lk, std::chrono::milliseconds(std::max(0, timeout_ms)), [this] { return !lines_.empty() || eof_; });
    if (!lines_.empty()) {
        line = std::move(lines_.front());
        lines_.pop_front();
        return 1;
    }
    return eof_ ? -1 : 0;
}

int run_capture(const std::vector<std::string>& cmd, const EnvList& env, std::string& out, int timeout_ms) {
    Process p;
    if (!p.start(cmd, env, nullptr)) return -1;
    p.close_input();
    auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    std::string line;
    for (;;) {
        int left = (int)std::chrono::duration_cast<std::chrono::milliseconds>(until - std::chrono::steady_clock::now()).count();
        if (left <= 0) {
            p.kill();
            return -1;
        }
        int r = p.read_line(line, left);
        if (r < 0) break;
        if (r > 0) out += line + "\n";
    }
    int code = p.wait(2000);
    return code == -1000 ? -1 : code;
}

}  // namespace kor
