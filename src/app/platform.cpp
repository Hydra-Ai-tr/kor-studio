#include "platform.h"

#include "i18n.h"

#include <cstdlib>
#include <filesystem>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <commdlg.h>
#include <mmsystem.h>
#include <shellapi.h>

#include <algorithm>
#include <cwchar>

static std::string narrow(const wchar_t* w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return "";
    std::string s((size_t)n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
    return s;
}

static std::string file_dialog(int kind, bool save) {
    wchar_t buf[4096] = L"";
    OPENFILENAMEW ofn;
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = GetActiveWindow();
    ofn.lpstrFile = buf;
    ofn.nMaxFile = 4096;
    ofn.lpstrFilter = kind == DLG_MODEL ? L"Kor model (*.kor)\0*.kor\0All files\0*.*\0"
                                        : L"Data (*.txt;*.csv;*.tsv;*.png;*.bmp;*.ppm;*.pgm;*.wav)\0*.txt;*.csv;*.tsv;*.png;*.bmp;*.ppm;*.pgm;*.wav\0"
                                          L"All files\0*.*\0";
    ofn.Flags = OFN_NOCHANGEDIR | (save ? OFN_OVERWRITEPROMPT : OFN_FILEMUSTEXIST);
    if (save) ofn.lpstrDefExt = L"kor";
    BOOL ok = save ? GetSaveFileNameW(&ofn) : GetOpenFileNameW(&ofn);
    return ok ? narrow(buf) : std::string();
}

std::string open_file_dialog(int kind) { return file_dialog(kind, false); }
std::string save_file_dialog() { return file_dialog(DLG_MODEL, true); }
bool has_native_dialogs() { return true; }

std::string exe_dir() {
    wchar_t buf[4096];
    DWORD n = GetModuleFileNameW(nullptr, buf, 4096);
    if (n == 0 || n >= 4096) return ".";
    return std::filesystem::path(buf).parent_path().u8string();
}

std::string config_dir() {
    const wchar_t* base = _wgetenv(L"APPDATA");
    std::filesystem::path p = base ? std::filesystem::path(base) / "KorStudio" : std::filesystem::path(".");
    std::error_code ec;
    std::filesystem::create_directories(p, ec);
    return p.u8string();
}

int system_language() {
    switch (GetUserDefaultUILanguage() & 0x3ff) {
        case 0x1f: return LANG_TR;
        case 0x19: return LANG_RU;
        default: return LANG_EN;
    }
}

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

int run_process(const std::vector<std::string>& args, const std::function<void(const std::string&)>& on_output,
                const std::atomic<bool>* cancel) {
    std::wstring cmd;
    for (auto& a : args) {
        if (!cmd.empty()) cmd += L' ';
        cmd += quote_arg(widen(a));
    }
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(sa);
    sa.lpSecurityDescriptor = nullptr;
    sa.bInheritHandle = TRUE;
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return -1;
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
    STARTUPINFOW si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = wr;
    si.hStdError = wr;
    si.hStdInput = nul;
    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));
    std::vector<wchar_t> buf(cmd.begin(), cmd.end());
    buf.push_back(L'\0');
    BOOL ok = CreateProcessW(nullptr, buf.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
    CloseHandle(wr);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!ok) {
        CloseHandle(rd);
        return -1;
    }
    char tmp[4096];
    bool exited = false, killed = false;
    for (;;) {
        DWORD avail = 0, n = 0;
        if (!PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr)) break;  // every writer has closed the pipe
        if (avail > 0) {
            if (!ReadFile(rd, tmp, std::min<DWORD>(avail, sizeof(tmp)), &n, nullptr) || n == 0) break;
            on_output(std::string(tmp, n));
            continue;
        }
        if (exited) break;  // the process is gone and its output is drained
        if (cancel && *cancel) {
            TerminateProcess(pi.hProcess, 1);
            killed = true;
            break;
        }
        // Processes started by the child may keep the pipe open, so wait on the child itself.
        exited = WaitForSingleObject(pi.hProcess, 25) == WAIT_OBJECT_0;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    if (killed) code = (DWORD)-2;
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(rd);
    return (int)code;
}

static bool python_works(const std::vector<std::string>& cmd, std::string& version) {
    std::vector<std::string> args = cmd;
    args.push_back("--version");
    std::string out;
    if (run_process(args, out) != 0 || out.compare(0, 8, "Python 3") != 0) return false;
    version = out.substr(0, out.find_first_of("\r\n"));
    return true;
}

std::vector<std::string> find_python(std::string& version) {
    namespace fs = std::filesystem;
    std::vector<std::vector<std::string>> candidates;
    // 1. Standard install folders (newest first). Checked before PATH because a
    //    fresh Windows has a "python.exe" that only opens the Microsoft Store.
    std::vector<fs::path> roots;
    if (const wchar_t* la = _wgetenv(L"LOCALAPPDATA")) roots.push_back(fs::path(la) / "Programs" / "Python");
    if (const wchar_t* pf = _wgetenv(L"ProgramFiles")) roots.push_back(fs::path(pf));
    roots.push_back(fs::path("C:\\"));
    for (auto& root : roots) {
        std::vector<fs::path> found;
        std::error_code ec;
        for (fs::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
            std::wstring name = it->path().filename().wstring();
            if (name.rfind(L"Python3", 0) == 0 && fs::exists(it->path() / "python.exe", ec)) found.push_back(it->path() / "python.exe");
        }
        std::sort(found.rbegin(), found.rend());
        for (auto& p : found) candidates.push_back({p.u8string()});
    }
    // 2. The py launcher, then python on PATH (skipping the Store alias).
    wchar_t buf[4096];
    if (SearchPathW(nullptr, L"py.exe", nullptr, 4096, buf, nullptr)) candidates.push_back({narrow(buf), "-3"});
    for (const wchar_t* exe : {L"python.exe", L"python3.exe"})
        if (SearchPathW(nullptr, exe, nullptr, 4096, buf, nullptr) && !wcsstr(buf, L"WindowsApps")) candidates.push_back({narrow(buf)});
    for (auto& c : candidates)
        if (python_works(c, version)) return c;
    return {};
}

void set_env(const char* name, const char* value) { _wputenv_s(widen(name).c_str(), widen(value).c_str()); }

void open_folder(const std::string& path) {
    ShellExecuteW(nullptr, L"open", widen(path).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

bool play_wav(const std::string& path) {
    return PlaySoundW(widen(path).c_str(), nullptr, SND_FILENAME | SND_ASYNC | SND_NODEFAULT) != FALSE;
}

void stop_sound() { PlaySoundW(nullptr, nullptr, 0); }

#else  // POSIX

#include <climits>
#include <csignal>
#include <cstdio>
#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

int run_process(const std::vector<std::string>& args, const std::function<void(const std::string&)>& on_output,
                const std::atomic<bool>* cancel) {
    if (args.empty()) return -1;
    int fds[2];
    if (pipe(fds) != 0) return -1;
    pid_t pid = fork();
    if (pid < 0) { close(fds[0]); close(fds[1]); return -1; }
    if (pid == 0) {
        int nul = open("/dev/null", O_RDONLY);
        if (nul >= 0) dup2(nul, 0);
        dup2(fds[1], 1);
        dup2(fds[1], 2);
        close(fds[0]);
        close(fds[1]);
        std::vector<char*> argv;
        for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
        argv.push_back(nullptr);
        execvp(argv[0], argv.data());
        _exit(127);
    }
    close(fds[1]);
    char tmp[4096];
    bool killed = false;
    for (;;) {
        pollfd pf{fds[0], POLLIN, 0};
        int r = poll(&pf, 1, 25);
        if (r > 0) {
            ssize_t n = read(fds[0], tmp, sizeof(tmp));
            if (n <= 0) break;
            on_output(std::string(tmp, (size_t)n));
        } else if (r < 0) {
            break;
        }
        if (cancel && *cancel) {
            kill(pid, SIGKILL);
            killed = true;
            break;
        }
    }
    close(fds[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    if (killed) return -2;
    return WIFEXITED(st) ? (WEXITSTATUS(st) == 127 ? -1 : WEXITSTATUS(st)) : -1;
}

std::vector<std::string> find_python(std::string& version) {
    for (const char* name : {"python3", "python"}) {
        std::string out;
        if (run_process({name, "--version"}, out) == 0 && out.compare(0, 8, "Python 3") == 0) {
            version = out.substr(0, out.find_first_of("\r\n"));
            return {name};
        }
    }
    return {};
}

void set_env(const char* name, const char* value) { setenv(name, value, 1); }

void open_folder(const std::string& path) {
#if defined(__APPLE__)
    std::string out;
    run_process({"open", path}, out);
#else
    std::string out;
    run_process({"xdg-open", path}, out);
#endif
}

static pid_t g_player = 0;

void stop_sound() {
    if (g_player > 0) {
        kill(g_player, SIGTERM);
        waitpid(g_player, nullptr, 0);
        g_player = 0;
    }
}

bool play_wav(const std::string& path) {
    stop_sound();
#if defined(__APPLE__)
    const char* player = "afplay";
#else
    const char* player = "aplay";
#endif
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        int nul = open("/dev/null", O_WRONLY);
        if (nul >= 0) { dup2(nul, 1); dup2(nul, 2); }
        execlp(player, player, path.c_str(), (char*)nullptr);
        _exit(127);
    }
    g_player = pid;
    return true;
}

std::string open_file_dialog(int) { return ""; }
std::string save_file_dialog() { return ""; }
bool has_native_dialogs() { return false; }

std::string exe_dir() {
#if defined(__linux__)
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        return std::filesystem::path(buf).parent_path().string();
    }
#endif
    return ".";
}

std::string config_dir() {
    const char* xdg = std::getenv("XDG_CONFIG_HOME");
    const char* home = std::getenv("HOME");
    std::filesystem::path p = xdg && *xdg ? std::filesystem::path(xdg)
                              : home ? std::filesystem::path(home) / ".config" : std::filesystem::path(".");
    p /= "korstudio";
    std::error_code ec;
    std::filesystem::create_directories(p, ec);
    return p.string();
}

int system_language() {
    const char* l = std::getenv("LANG");
    std::string s = l ? l : "";
    if (s.rfind("tr", 0) == 0) return LANG_TR;
    if (s.rfind("ru", 0) == 0) return LANG_RU;
    return LANG_EN;
}

#endif

int run_process(const std::vector<std::string>& args, std::string& output) {
    return run_process(args, [&](const std::string& s) { output += s; }, nullptr);
}
