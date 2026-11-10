// Kor Studio - application UI (Dear ImGui).
#pragma once

struct ImFont;

namespace app {

enum Page { PAGE_HOME = 0, PAGE_DATA, PAGE_MODEL, PAGE_TRAIN, PAGE_TEST, PAGE_FILES, PAGE_PLUGINS, PAGE_COUNT };

void init(float ui_scale, ImFont* title_font);
void frame();      // draws one frame of the whole interface
void shutdown();   // stops training and saves settings

// Scripting hooks used by command-line options.
void set_page(int page);
bool load_data(const char* utf8_path);
void start_training();
void set_train_tab(int tab);  // 0 chart, 1 log, 2 question tests

}  // namespace app
