// Kor Studio - design system on top of Dear ImGui: palette, fonts, vector
// icons and the custom widgets every page is built from.
#pragma once
#include "imgui.h"

#include <string>
#include <vector>

namespace kit {

extern float sc;  // UI scale (monitor content scale)
inline float px(float v) { return v * sc; }

struct Fonts {
    ImFont* body = nullptr;        // regular text
    ImFont* bold = nullptr;        // semibold: buttons, card titles
    ImFont* small = nullptr;       // captions, labels
    ImFont* small_bold = nullptr;  // badges, table headers, section labels
    ImFont* h1 = nullptr;          // page titles
    ImFont* h2 = nullptr;          // section titles
    ImFont* stat = nullptr;        // big numbers
    ImFont* mono = nullptr;        // logs, previews, consoles
    ImFont* hero = nullptr;        // the big title on the home page
};
extern Fonts F;

namespace C {
inline const ImVec4 bg(0.051f, 0.043f, 0.078f, 1);
inline const ImVec4 sidebar(0.039f, 0.031f, 0.063f, 1);
inline const ImVec4 inset(0.043f, 0.035f, 0.067f, 1);
inline const ImVec4 surface(0.078f, 0.067f, 0.118f, 1);
inline const ImVec4 surface2(0.106f, 0.090f, 0.157f, 1);
inline const ImVec4 surface3(0.141f, 0.122f, 0.208f, 1);
inline const ImVec4 border(0.149f, 0.129f, 0.212f, 1);
inline const ImVec4 border_hi(0.220f, 0.192f, 0.310f, 1);
inline const ImVec4 text(0.925f, 0.918f, 0.957f, 1);
inline const ImVec4 muted(0.580f, 0.557f, 0.678f, 1);
inline const ImVec4 subtle(0.384f, 0.361f, 0.478f, 1);
inline const ImVec4 accent(0.545f, 0.361f, 0.965f, 1);
inline const ImVec4 accent_hi(0.690f, 0.561f, 1.000f, 1);
inline const ImVec4 accent_lo(0.427f, 0.247f, 0.851f, 1);
inline const ImVec4 on_accent(1.000f, 1.000f, 1.000f, 1);
inline const ImVec4 good(0.204f, 0.780f, 0.482f, 1);
inline const ImVec4 bad(0.949f, 0.333f, 0.353f, 1);
inline const ImVec4 warn(0.961f, 0.710f, 0.267f, 1);
inline const ImVec4 blue(0.302f, 0.639f, 1.000f, 1);
inline const ImVec4 violet(0.910f, 0.475f, 0.976f, 1);
inline const ImVec4 teal(0.180f, 0.769f, 0.714f, 1);
}  // namespace C

// Colour for the draw list; honours the style alpha, so disabled widgets fade too.
ImU32 col(const ImVec4& c, float alpha = 1.f);
ImVec4 with_alpha(const ImVec4& c, float a);
ImVec4 mix(const ImVec4& a, const ImVec4& b, float t);

void load_fonts(float scale);  // before the first frame
void apply_style();

struct FontScope {
    explicit FontScope(ImFont* f) : on_(f != nullptr) { if (on_) ImGui::PushFont(f); }
    ~FontScope() { if (on_) ImGui::PopFont(); }
    FontScope(const FontScope&) = delete;
    FontScope& operator=(const FontScope&) = delete;
private:
    bool on_;
};

ImVec2 text_size(ImFont* f, const char* s, const char* end = nullptr, float wrap = -1.f);
// Draws one line, cut with an ellipsis if wider than max_w. Returns true if it was cut.
bool text_ellipsis(ImDrawList* dl, ImFont* f, ImVec2 pos, float max_w, const char* s, ImU32 color);
float anim(ImGuiID id, bool on, float speed = 14.f);  // 0..1, eases towards on

enum Icon {
    IC_NONE = 0, IC_HOME, IC_DATA, IC_MODEL, IC_TRAIN, IC_CHAT, IC_FILES, IC_PLUGIN,
    IC_CPU, IC_MEMORY, IC_BOLT, IC_GPU, IC_THREADS, IC_CHECK, IC_X, IC_FOLDER, IC_SAVE,
    IC_UPLOAD, IC_REFRESH, IC_COPY, IC_SPARK, IC_CLOCK, IC_GAUGE, IC_TARGET, IC_TREND,
    IC_PLAY, IC_PAUSE, IC_STOP, IC_INFO, IC_WARN, IC_DOC, IC_TABLE, IC_CHEVRON_DOWN,
    IC_CHEVRON_RIGHT, IC_GLOBE, IC_PLUS, IC_ARROW_UP, IC_TRASH, IC_LAYERS, IC_PERCENT,
    IC_CHATS, IC_BELL, IC_CLOUD, IC_GEAR, IC_KEYBOARD, IC_WAVE, IC_FLASK, IC_CUBE, IC_CODE,
    IC_PLAY_O, IC_ARROW_RIGHT, IC_MINUS, IC_MAXIMIZE, IC_RESTORE, IC_SIDEBAR, IC_TERMINAL, IC_DISK,
};
void icon(ImDrawList* dl, Icon ic, ImVec2 center, float size, ImU32 color);
void draw_logo(ImDrawList* dl, ImVec2 p, float size);

// ---- text
void text(const char* s, const ImVec4& c);
void text_wrapped(const char* s, const ImVec4& c, ImFont* f = nullptr);
void section_title(const char* s);
void tooltip(const char* s);  // wrapped tooltip for the last item
void gap(float h = 6.f);

// ---- layout
// Page title and subtitle. With actions_w > 0 the cursor is left at the
// top right so the caller can place that many pixels of buttons there.
void header(const char* title, const char* subtitle, float actions_w = 0);
void header_end();
int fit_columns(float min_w, int max_cols, float gap);
float column_width(int cols, float gap);

// Cards are auto-height child windows. Pass min_h from an EqualRow to give a
// row of cards the same height; end_card() reports the natural height back.
void begin_card(const char* id, const char* title = nullptr, float min_h = 0, float width = 0, const char* subtitle = nullptr);
void end_card(float* natural_h = nullptr);
// True (once) if a card was shown for the first time this frame; the frame should then be
// built again before it is presented, so the card is never drawn while still unmeasured.
bool take_relayout();
bool begin_collapsible_card(const char* id, const char* title, bool* open, const char* summary);
struct EqualRow {
    float prev = 0, cur = 0;
    float* track() { return &cur; }
    void next() { prev = cur; cur = 0; }
};
float card_padding_x();
float card_padding_y();
// Unscaled card padding for the cards begun while this is alive.
struct CardPadding {
    CardPadding(float x, float y);
    ~CardPadding();
    float old_x, old_y;
};

// ---- controls
enum ButtonKind { BTN_PRIMARY, BTN_SECONDARY, BTN_GHOST, BTN_DANGER };
bool button(const char* label, ButtonKind kind = BTN_SECONDARY, Icon ic = IC_NONE, float width = 0, float height = 0);
float button_width(const char* label, Icon ic = IC_NONE);
bool segmented(const char* id, int* value, const char* const* labels, int n);
bool tabs(const char* id, int* value, const char* const* labels, int n);
bool nav_item(const char* label, Icon ic, bool selected, int status);  // status: 0 none, 1 done, 2 running
void badge(const char* text, const ImVec4& c, bool dot = true, bool pulse = false);
float badge_width(const char* text, bool dot = true);
void progress_bar(float frac, float height, const ImVec4& c);
// Track-and-knob sliders with the value in a box on the right; width comes from SetNextItemWidth.
bool slider_int(const char* id, int* v, int lo, int hi, const char* fmt = "%d");
bool slider_float(const char* id, float* v, float lo, float hi, const char* fmt = "%.2f");
bool checkbox(const char* label, bool* v);
// On/off switch without a label.
bool toggle(const char* id, bool* v);
float toggle_width();
// BeginCombo with a thin chevron instead of the arrow button; pair with ImGui::EndCombo.
bool begin_combo(const char* id, const char* preview);
bool combo(const char* id, int* current, const char* const* items, int n);
void ring(float frac, float radius, float thickness, const ImVec4& c, const char* center);
// Label in a fixed-width column to the left of the next widget, which gets the rest of the row.
void field_label(const char* label, float label_w, const char* help = nullptr);
void banner(const char* text, const ImVec4& c, Icon ic);
// Centered icon, title, text and up to two buttons. Returns 1 or 2 for a click.
int empty_state(Icon ic, const char* title, const char* text, const char* button1 = nullptr, const char* button2 = nullptr);

struct Stat {
    std::string label, value, sub;
    ImVec4 color = C::text;
    Icon ic = IC_NONE;
};
float stat_tile_height(bool big, bool sub);
void stat_tile(const Stat& s, float width, float height, bool big);
void stat_grid(const std::vector<Stat>& stats, float min_w, bool big);

}  // namespace kit
