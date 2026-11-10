// Kor Studio - small OS-specific helpers. All strings are UTF-8.
#pragma once
#include <atomic>
#include <functional>
#include <string>
#include <vector>

enum DialogKind { DLG_DATA = 0, DLG_MODEL = 1 };

// Native "open file" / "save file" dialogs. Return "" if cancelled or if
// the platform has no native dialog (the user can type a path instead).
std::string open_file_dialog(int kind);
std::string save_file_dialog();
bool has_native_dialogs();

std::string exe_dir();
std::string config_dir();   // per-user settings directory (created on demand)
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
void open_folder(const std::string& path);  // shows a folder in the file manager
// Plays a WAV file in the background; a new sound replaces the one playing.
bool play_wav(const std::string& path);
void stop_sound();
