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
                                        : L"Text and tables (*.txt;*.csv;*.tsv)\0*.txt;*.csv;*.tsv\0All files\0*.*\0";
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

#else  // POSIX

#include <climits>
#include <unistd.h>

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
