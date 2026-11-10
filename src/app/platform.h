// Kor Studio - small OS-specific helpers. All strings are UTF-8.
#pragma once
#include <string>

enum DialogKind { DLG_DATA = 0, DLG_MODEL = 1 };

// Native "open file" / "save file" dialogs. Return "" if cancelled or if
// the platform has no native dialog (the user can type a path instead).
std::string open_file_dialog(int kind);
std::string save_file_dialog();
bool has_native_dialogs();

std::string exe_dir();
std::string config_dir();   // per-user settings directory (created on demand)
int system_language();      // Lang value guessed from the OS locale
