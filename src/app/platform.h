// Kor Studio - small OS-specific helpers. All strings are UTF-8.
#pragma once
#include <atomic>
#include <functional>
#include <string>
#include <vector>

enum DialogKind { DLG_DATA = 0, DLG_MODEL = 1, DLG_PROJECT = 2, DLG_CHAT_MODEL = 3, DLG_GGUF = 4 };

// Native "open file" / "save file" dialogs. Return "" if cancelled or if
// the platform has no native dialog (the user can type a path instead).
std::string open_file_dialog(int kind);
std::string save_file_dialog();
bool has_native_dialogs();

std::string exe_dir();
std::string config_dir();   // per-user settings directory (created on demand)
std::string documents_dir();  // the user's documents folder (home if there is none)

// Whole-machine load: CPU busy share 0..1 since the previous call (-1 on the
// first call or if unknown) and memory in use / installed, in GB.
// GPU memory is that of the largest graphics adapter (its shared memory for
// integrated graphics); gpu_used_gb is -1 where it cannot be read.
struct Usage {
    float cpu = -1;
    double ram_used_gb = 0, ram_total_gb = 0;
    double gpu_used_gb = -1, gpu_total_gb = 0;
};
Usage sample_usage();
std::string user_name();  // the account the app runs under
// Opens a terminal in `dir` where kor-cli (next to the app) can be run.
void open_terminal(const std::string& dir);
// Runs a command line in a new terminal window that stays open, in `dir`.
void run_in_terminal(const std::string& dir, const std::vector<std::string>& args);
int system_language();      // Lang value guessed from the OS locale

// Runs a program (args[0]) with arguments and waits for it. Standard output
// and error are passed to `on_output` as they arrive. Setting `*cancel` kills
// the program. Returns the exit code, -1 if the program could not be started,
// or -2 if it was cancelled. No console window is shown.
int run_process(const std::vector<std::string>& args, const std::function<void(const std::string&)>& on_output,
                const std::atomic<bool>* cancel);
// Same, collecting all output into `output`.
int run_process(const std::vector<std::string>& args, std::string& output);
// Finds a working Python 3 interpreter. Returns the command (possibly with
// leading arguments, e.g. {"py", "-3"}) or an empty vector; `version` gets
// its "Python 3.x.y" banner.
std::vector<std::string> find_python(std::string& version);
void set_env(const char* name, const char* value);
// Downloads a small text resource over HTTP(S). Blocking; call it off the UI thread.
bool http_get(const std::string& url, std::string& body, std::string& err);
void open_folder(const std::string& path);  // shows a folder in the file manager
// Plays a WAV file in the background; a new sound replaces the one playing.
bool play_wav(const std::string& path);
void stop_sound();
