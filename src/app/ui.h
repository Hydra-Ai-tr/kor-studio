// Kor Studio - application UI (Dear ImGui).
#pragma once

namespace app {

enum Page { PAGE_HOME = 0, PAGE_DATA, PAGE_MODEL, PAGE_TRAIN, PAGE_TEST, PAGE_FILES, PAGE_PLUGINS, PAGE_COUNT };

void init(float ui_scale);  // loads fonts and the theme; call before the first frame
void frame();      // draws one frame of the whole interface
void shutdown();   // stops training and saves settings

// Scripting hooks used by command-line options.
void set_page(int page);
bool load_data(const char* utf8_path);  // text, table, image folder or WAV, by the path
bool load_model(const char* utf8_path);
void generate_media();                   // playground: new pictures or sound from the model
void start_training();
void set_train_tab(int tab);  // 0 chart, 1 log, 2 question tests, 3 skills
// Scripted runs: use exactly these skills (plugin file names) and leave the settings file alone.
void use_skills_for_session(const char* const* files, int n);
void keep_settings();
void scroll_to_end();          // scrolls the page to the bottom once it is drawn
void load_community_list();    // fetches the community plugin list once Python is found

}  // namespace app
