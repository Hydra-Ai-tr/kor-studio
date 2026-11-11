// Kor Studio - application UI (Dear ImGui).
#pragma once

namespace app {

enum Page { PAGE_HOME = 0, PAGE_PROJECTS, PAGE_DATA, PAGE_MODEL, PAGE_TRAIN, PAGE_TEST, PAGE_CHAT, PAGE_FILES, PAGE_PLUGINS, PAGE_COUNT };

void init(float ui_scale);  // loads fonts and the theme; call before the first frame
void frame();      // draws one frame of the whole interface
void shutdown();   // stops training and saves settings

// Scripting hooks used by command-line options.
void set_page(int page);
bool load_data(const char* utf8_path);  // text, table, image folder or WAV, by the path
bool load_model(const char* utf8_path);
bool chat_load(const char* utf8_path);   // opens a model on the chat page
bool open_project(const char* utf8_path);  // a project folder or its project.korproj
bool save_project_to(const char* utf8_dir);  // saves the current work there as a project and opens it
void chat_say(const char* utf8_text);    // sends a chat message
void generate_media();                   // playground: new pictures or sound from the model
void start_training();
// Scripted runs: number of steps (> 0) and the graphics card (0 off, 1 on, -1 as set).
void set_train_options(int steps, int gpu);
void set_train_tab(int tab);  // 0 chart, 1 log, 2 question tests, 3 skills
// Scripted runs: use exactly these skills (plugin file names) and leave the settings file alone.
void use_skills_for_session(const char* const* files, int n);
void keep_settings();
void scroll_to_end();          // scrolls the page to the bottom once it is drawn
void load_community_list();    // fetches the community plugin list once Python is found
void open_settings();          // shows the settings window
void check_updates();          // asks GitHub for the latest release now
void open_clean();             // shows the cleaning window for the loaded text

// The window frame drawn by the app (Windows): the title bar with the window
// buttons. Its empty parts move the window.
enum WindowCommand { WIN_NONE = 0, WIN_MINIMIZE, WIN_MAXIMIZE, WIN_CLOSE };
void set_custom_frame(bool on, bool maximized);
bool title_bar_drag_at(float x, float y);  // window coordinates, in pixels
int take_window_command();

}  // namespace app
