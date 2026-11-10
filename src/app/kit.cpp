#include "kit.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>

namespace kit {

float sc = 1.f;
Fonts F;

namespace {

const float kPi = 3.14159265f;

// Latin (incl. Turkish), Cyrillic and common punctuation (dashes, quotes, bullet, ellipsis).
const ImWchar kGlyphRanges[] = {0x0020, 0x00FF, 0x0100, 0x017F, 0x0400, 0x052F, 0x2010, 0x205E, 0};

const char* first_existing(std::initializer_list<const char*> paths) {
    for (const char* p : paths) {
        std::error_code ec;
        if (std::filesystem::exists(p, ec)) return p;
    }
    return nullptr;
}

ImFont* add_font(const char* path, float size) {
    if (!path) return nullptr;
    ImFontConfig cfg;
    cfg.OversampleH = 3;
    cfg.OversampleV = 1;
    return ImGui::GetIO().Fonts->AddFontFromFileTTF(path, size, &cfg, kGlyphRanges);
}

const char* label_end(const char* s) {
    const char* e = std::strstr(s, "##");
    return e ? e : s + std::strlen(s);
}

int utf8_len(unsigned char c) { return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4; }

// header() state
ImVec2 g_hdr_start;
float g_hdr_bottom = 0;

}  // namespace

ImU32 col(const ImVec4& c, float alpha) { return ImGui::GetColorU32(ImVec4(c.x, c.y, c.z, c.w * alpha)); }
ImVec4 with_alpha(const ImVec4& c, float a) { return ImVec4(c.x, c.y, c.z, a); }
ImVec4 mix(const ImVec4& a, const ImVec4& b, float t) {
    return ImVec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t);
}

void load_fonts(float scale) {
    sc = scale;
    ImGuiIO& io = ImGui::GetIO();
    const char* regular = first_existing({
        "C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf"});
    const char* semibold = first_existing({
        "C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/segoeuib.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-SemiBold.ttf", "/usr/share/fonts/noto/NotoSans-SemiBold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"});
    const char* mono = first_existing({
        "C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf"});
    if (!regular) {
        io.Fonts->AddFontDefault();  // ASCII only: Turkish/Russian text needs a system font
        return;
    }
    if (!semibold) semibold = regular;
    if (!mono) mono = regular;
    F.body = add_font(regular, 16.f * scale);  // first font = default
    F.bold = add_font(semibold, 16.f * scale);
    F.small = add_font(regular, 13.5f * scale);
    F.small_bold = add_font(semibold, 13.f * scale);
    F.h1 = add_font(semibold, 26.f * scale);
    F.h2 = add_font(semibold, 18.f * scale);
    F.stat = add_font(semibold, 24.f * scale);
    F.mono = add_font(mono, 14.5f * scale);
}

void apply_style() {
    ImGuiStyle& st = ImGui::GetStyle();
    st = ImGuiStyle();
    ImGui::StyleColorsDark(&st);
    st.WindowPadding = ImVec2(12, 10);
    st.FramePadding = ImVec2(11, 7);
    st.ItemSpacing = ImVec2(10, 10);
    st.ItemInnerSpacing = ImVec2(8, 6);
    st.CellPadding = ImVec2(10, 7);
    st.ScrollbarSize = 10;
    st.GrabMinSize = 12;
    st.WindowRounding = 0;
    st.ChildRounding = 12;
    st.FrameRounding = 8;
    st.PopupRounding = 10;
    st.ScrollbarRounding = 10;
    st.GrabRounding = 6;
    st.TabRounding = 8;
    st.WindowBorderSize = 0;
    st.ChildBorderSize = 1;
    st.PopupBorderSize = 1;
    st.FrameBorderSize = 1;
    st.DisabledAlpha = 0.45f;
    st.SelectableTextAlign = ImVec2(0, 0.5f);
    st.ScaleAllSizes(sc);

    ImVec4* c = st.Colors;
    c[ImGuiCol_Text] = C::text;
    c[ImGuiCol_TextDisabled] = C::subtle;
    c[ImGuiCol_WindowBg] = C::bg;
    c[ImGuiCol_ChildBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_PopupBg] = C::surface2;
    c[ImGuiCol_Border] = C::border;
    c[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_FrameBg] = C::surface2;
    c[ImGuiCol_FrameBgHovered] = C::surface3;
    c[ImGuiCol_FrameBgActive] = mix(C::surface3, C::border_hi, 0.5f);
    c[ImGuiCol_TitleBg] = c[ImGuiCol_TitleBgActive] = c[ImGuiCol_TitleBgCollapsed] = C::sidebar;
    c[ImGuiCol_MenuBarBg] = C::sidebar;
    c[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_ScrollbarGrab] = C::border_hi;
    c[ImGuiCol_ScrollbarGrabHovered] = mix(C::border_hi, C::muted, 0.35f);
    c[ImGuiCol_ScrollbarGrabActive] = C::muted;
    c[ImGuiCol_CheckMark] = C::accent;
    c[ImGuiCol_SliderGrab] = C::accent;
    c[ImGuiCol_SliderGrabActive] = C::accent_hi;
    c[ImGuiCol_Button] = C::surface2;
    c[ImGuiCol_ButtonHovered] = C::surface3;
    c[ImGuiCol_ButtonActive] = C::border_hi;
    c[ImGuiCol_Header] = with_alpha(C::accent, 0.16f);
    c[ImGuiCol_HeaderHovered] = C::surface3;
    c[ImGuiCol_HeaderActive] = with_alpha(C::accent, 0.24f);
    c[ImGuiCol_Separator] = c[ImGuiCol_SeparatorHovered] = c[ImGuiCol_SeparatorActive] = C::border;
    c[ImGuiCol_ResizeGrip] = c[ImGuiCol_ResizeGripHovered] = c[ImGuiCol_ResizeGripActive] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_Tab] = C::surface2;
    c[ImGuiCol_TabHovered] = C::surface3;
    c[ImGuiCol_TabSelected] = C::surface3;
    c[ImGuiCol_PlotLines] = c[ImGuiCol_PlotHistogram] = C::accent;
    c[ImGuiCol_PlotLinesHovered] = c[ImGuiCol_PlotHistogramHovered] = C::accent_hi;
    c[ImGuiCol_TableHeaderBg] = C::surface2;
    c[ImGuiCol_TableBorderStrong] = C::border;
    c[ImGuiCol_TableBorderLight] = with_alpha(C::border, 0.7f);
    c[ImGuiCol_TableRowBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_TableRowBgAlt] = ImVec4(1, 1, 1, 0.018f);
    c[ImGuiCol_TextSelectedBg] = with_alpha(C::accent, 0.35f);
    c[ImGuiCol_ModalWindowDimBg] = ImVec4(0, 0, 0, 0.5f);
}

ImVec2 text_size(ImFont* f, const char* s, const char* end, float wrap) {
    if (!f) f = ImGui::GetFont();
    ImVec2 r = f->CalcTextSizeA(f->FontSize, FLT_MAX, wrap, s, end);
    return ImVec2(std::ceil(r.x), r.y);
}

bool text_ellipsis(ImDrawList* dl, ImFont* f, ImVec2 pos, float max_w, const char* s, ImU32 color) {
    if (!f) f = ImGui::GetFont();
    const float fs = f->FontSize;
    if (f->CalcTextSizeA(fs, FLT_MAX, 0, s).x <= max_w) {
        dl->AddText(f, fs, pos, color, s);
        return false;
    }
    const char* ell = "\xE2\x80\xA6";
    float ew = f->CalcTextSizeA(fs, FLT_MAX, 0, ell).x, w = 0;
    const char* p = s;
    while (*p) {
        const char* q = p + utf8_len((unsigned char)*p);
        float cw = f->CalcTextSizeA(fs, FLT_MAX, 0, p, q).x;
        if (w + cw + ew > max_w) break;
        w += cw;
        p = q;
    }
    dl->AddText(f, fs, pos, color, s, p);
    dl->AddText(f, fs, ImVec2(pos.x + w, pos.y), color, ell);
    return true;
}

float anim(ImGuiID id, bool on, float speed) {
    float* v = ImGui::GetStateStorage()->GetFloatRef(id ^ 0x6B6F7231u, 0.f);
    float target = on ? 1.f : 0.f;
    *v += (target - *v) * std::min(1.f, ImGui::GetIO().DeltaTime * speed);
    if (std::fabs(*v - target) < 0.002f) *v = target;
    return *v;
}

// ------------------------------------------------------------------ icons

void icon(ImDrawList* dl, Icon ic, ImVec2 c, float s, ImU32 color) {
    const float t = std::max(1.4f, s * 0.085f);
    auto P = [&](float x, float y) { return ImVec2(c.x + x * s, c.y + y * s); };
    auto line = [&](float x0, float y0, float x1, float y1) { dl->AddLine(P(x0, y0), P(x1, y1), color, t); };
    auto poly = [&](std::initializer_list<ImVec2> pts, bool closed) {
        for (const ImVec2& q : pts) dl->PathLineTo(P(q.x, q.y));
        dl->PathStroke(color, closed ? ImDrawFlags_Closed : 0, t);
    };
    auto rect = [&](float x0, float y0, float x1, float y1, float r) { dl->AddRect(P(x0, y0), P(x1, y1), color, r * s, 0, t); };
    auto circle = [&](float x, float y, float r) { dl->AddCircle(P(x, y), r * s, color, 0, t); };
    auto dot = [&](float x, float y, float r) { dl->AddCircleFilled(P(x, y), r * s, color); };
    switch (ic) {
        case IC_NONE: break;
        case IC_HOME:
            poly({{-0.42f, 0.0f}, {0.f, -0.38f}, {0.42f, 0.0f}}, false);
            poly({{-0.3f, -0.1f}, {-0.3f, 0.38f}, {0.3f, 0.38f}, {0.3f, -0.1f}}, false);
            poly({{-0.08f, 0.38f}, {-0.08f, 0.14f}, {0.08f, 0.14f}, {0.08f, 0.38f}}, false);
            break;
        case IC_DATA:
            dl->AddEllipse(P(0, -0.27f), ImVec2(0.36f * s, 0.13f * s), color, 0, 0, t);
            line(-0.36f, -0.27f, -0.36f, 0.27f);
            line(0.36f, -0.27f, 0.36f, 0.27f);
            dl->PathEllipticalArcTo(P(0, 0.0f), ImVec2(0.36f * s, 0.13f * s), 0, 0, kPi, 16);
            dl->PathStroke(color, 0, t);
            dl->PathEllipticalArcTo(P(0, 0.27f), ImVec2(0.36f * s, 0.13f * s), 0, 0, kPi, 16);
            dl->PathStroke(color, 0, t);
            break;
        case IC_MODEL:
            line(-0.3f, -0.26f, 0.3f, 0.f);
            line(-0.3f, 0.26f, 0.3f, 0.f);
            line(-0.3f, -0.26f, -0.3f, 0.26f);
            dot(-0.3f, -0.26f, 0.12f);
            dot(-0.3f, 0.26f, 0.12f);
            dot(0.3f, 0.f, 0.13f);
            break;
        case IC_TRAIN:
            poly({{-0.45f, 0.05f}, {-0.22f, 0.05f}, {-0.1f, -0.32f}, {0.08f, 0.36f}, {0.2f, 0.05f}, {0.45f, 0.05f}}, false);
            break;
        case IC_CHAT:
            rect(-0.42f, -0.34f, 0.42f, 0.2f, 0.12f);
            poly({{-0.18f, 0.2f}, {-0.26f, 0.4f}, {0.04f, 0.2f}}, false);
            break;
        case IC_FILES:
        case IC_FOLDER:
            poly({{-0.42f, -0.3f}, {-0.12f, -0.3f}, {0.f, -0.18f}, {0.42f, -0.18f}, {0.42f, 0.32f}, {-0.42f, 0.32f}}, true);
            break;
        case IC_PLUGIN:
            line(-0.14f, -0.45f, -0.14f, -0.24f);
            line(0.14f, -0.45f, 0.14f, -0.24f);
            rect(-0.3f, -0.24f, 0.3f, 0.1f, 0.08f);
            line(0.f, 0.1f, 0.f, 0.45f);
            break;
        case IC_CPU:
            rect(-0.26f, -0.26f, 0.26f, 0.26f, 0.06f);
            rect(-0.1f, -0.1f, 0.1f, 0.1f, 0.02f);
            for (float k : {-0.13f, 0.f, 0.13f}) {
                line(k, -0.26f, k, -0.42f); line(k, 0.26f, k, 0.42f);
                line(-0.26f, k, -0.42f, k); line(0.26f, k, 0.42f, k);
            }
            break;
        case IC_MEMORY:
            rect(-0.44f, -0.2f, 0.44f, 0.16f, 0.04f);
            for (float k : {-0.28f, -0.09f, 0.09f, 0.28f}) line(k, 0.16f, k, 0.32f);
            rect(-0.28f, -0.08f, -0.1f, 0.04f, 0.0f);
            rect(0.1f, -0.08f, 0.28f, 0.04f, 0.0f);
            break;
        case IC_BOLT:
            poly({{0.08f, -0.45f}, {-0.28f, 0.06f}, {-0.02f, 0.06f}, {-0.1f, 0.45f}, {0.28f, -0.08f}, {0.02f, -0.08f}}, true);
            break;
        case IC_GPU:
            rect(-0.44f, -0.24f, 0.44f, 0.2f, 0.06f);
            circle(0.1f, -0.02f, 0.13f);
            line(-0.3f, 0.2f, -0.3f, 0.36f);
            line(-0.26f, -0.1f, -0.16f, -0.1f);
            line(-0.26f, 0.06f, -0.16f, 0.06f);
            break;
        case IC_THREADS:
            line(-0.4f, -0.26f, 0.4f, -0.26f);
            line(-0.4f, 0.f, 0.25f, 0.f);
            line(-0.4f, 0.26f, 0.1f, 0.26f);
            break;
        case IC_CHECK:
            poly({{-0.32f, 0.02f}, {-0.09f, 0.25f}, {0.36f, -0.24f}}, false);
            break;
        case IC_X:
            line(-0.28f, -0.28f, 0.28f, 0.28f);
            line(0.28f, -0.28f, -0.28f, 0.28f);
            break;
        case IC_SAVE:
            line(0.f, -0.42f, 0.f, 0.14f);
            poly({{-0.18f, -0.04f}, {0.f, 0.14f}, {0.18f, -0.04f}}, false);
            poly({{-0.4f, 0.1f}, {-0.4f, 0.38f}, {0.4f, 0.38f}, {0.4f, 0.1f}}, false);
            break;
        case IC_UPLOAD:
            line(0.f, 0.14f, 0.f, -0.42f);
            poly({{-0.18f, -0.24f}, {0.f, -0.42f}, {0.18f, -0.24f}}, false);
            poly({{-0.4f, 0.1f}, {-0.4f, 0.38f}, {0.4f, 0.38f}, {0.4f, 0.1f}}, false);
            break;
        case IC_REFRESH: {
            dl->PathArcTo(c, 0.34f * s, -kPi * 0.35f, kPi * 1.35f, 24);
            dl->PathStroke(color, 0, t);
            ImVec2 e(c.x + std::cos(-kPi * 0.35f) * 0.34f * s, c.y + std::sin(-kPi * 0.35f) * 0.34f * s);
            dl->AddTriangleFilled(ImVec2(e.x + 0.16f * s, e.y - 0.02f * s), ImVec2(e.x - 0.06f * s, e.y - 0.16f * s),
                                  ImVec2(e.x - 0.02f * s, e.y + 0.1f * s), color);
            break;
        }
        case IC_COPY:
            rect(-0.36f, -0.36f, 0.14f, 0.14f, 0.06f);
            poly({{-0.14f, 0.24f}, {-0.14f, 0.36f}, {0.36f, 0.36f}, {0.36f, -0.14f}, {0.24f, -0.14f}}, false);
            break;
        case IC_SPARK:
            poly({{0.f, -0.45f}, {0.11f, -0.11f}, {0.45f, 0.f}, {0.11f, 0.11f}, {0.f, 0.45f}, {-0.11f, 0.11f}, {-0.45f, 0.f}, {-0.11f, -0.11f}}, true);
            break;
        case IC_CLOCK:
            circle(0, 0, 0.4f);
            line(0, 0, 0, -0.22f);
            line(0, 0, 0.16f, 0.1f);
            break;
        case IC_GAUGE:
            dl->PathArcTo(P(0, 0.12f), 0.4f * s, kPi, 2 * kPi, 20);
            dl->PathStroke(color, 0, t);
            line(0, 0.12f, 0.2f, -0.12f);
            dot(0, 0.12f, 0.07f);
            break;
        case IC_TARGET:
            circle(0, 0, 0.4f);
            circle(0, 0, 0.2f);
            dot(0, 0, 0.06f);
            break;
        case IC_TREND:
            poly({{-0.42f, -0.28f}, {-0.12f, 0.02f}, {0.06f, -0.12f}, {0.4f, 0.26f}}, false);
            poly({{0.16f, 0.28f}, {0.4f, 0.28f}, {0.4f, 0.04f}}, false);
            break;
        case IC_PLAY:
            dl->AddTriangleFilled(P(-0.24f, -0.34f), P(0.34f, 0.f), P(-0.24f, 0.34f), color);
            break;
        case IC_PAUSE:
            dl->AddRectFilled(P(-0.26f, -0.32f), P(-0.08f, 0.32f), color, 0.04f * s);
            dl->AddRectFilled(P(0.08f, -0.32f), P(0.26f, 0.32f), color, 0.04f * s);
            break;
        case IC_STOP:
            dl->AddRectFilled(P(-0.26f, -0.26f), P(0.26f, 0.26f), color, 0.06f * s);
            break;
        case IC_INFO:
            circle(0, 0, 0.4f);
            line(0, -0.04f, 0, 0.2f);
            dot(0, -0.18f, 0.05f);
            break;
        case IC_WARN:
            poly({{0.f, -0.4f}, {0.42f, 0.34f}, {-0.42f, 0.34f}}, true);
            line(0, -0.12f, 0, 0.1f);
            dot(0, 0.22f, 0.045f);
            break;
        case IC_DOC:
            poly({{-0.3f, -0.42f}, {0.12f, -0.42f}, {0.3f, -0.24f}, {0.3f, 0.42f}, {-0.3f, 0.42f}}, true);
            line(-0.16f, -0.04f, 0.16f, -0.04f);
            line(-0.16f, 0.14f, 0.16f, 0.14f);
            line(-0.16f, 0.3f, 0.06f, 0.3f);
            break;
        case IC_TABLE:
            rect(-0.4f, -0.34f, 0.4f, 0.34f, 0.06f);
            line(-0.4f, -0.1f, 0.4f, -0.1f);
            line(-0.4f, 0.12f, 0.4f, 0.12f);
            line(-0.1f, -0.34f, -0.1f, 0.34f);
            break;
        case IC_CHEVRON_DOWN:
            poly({{-0.26f, -0.12f}, {0.f, 0.14f}, {0.26f, -0.12f}}, false);
            break;
        case IC_CHEVRON_RIGHT:
            poly({{-0.12f, -0.26f}, {0.14f, 0.f}, {-0.12f, 0.26f}}, false);
            break;
        case IC_GLOBE:
            circle(0, 0, 0.4f);
            dl->AddEllipse(c, ImVec2(0.17f * s, 0.4f * s), color, 0, 0, t);
            line(-0.4f, 0, 0.4f, 0);
            break;
        case IC_PLUS:
            line(-0.3f, 0, 0.3f, 0);
            line(0, -0.3f, 0, 0.3f);
            break;
        case IC_ARROW_UP:
            line(0, 0.34f, 0, -0.32f);
            poly({{-0.24f, -0.08f}, {0.f, -0.32f}, {0.24f, -0.08f}}, false);
            break;
        case IC_TRASH:
            line(-0.38f, -0.26f, 0.38f, -0.26f);
            poly({{-0.12f, -0.26f}, {-0.12f, -0.4f}, {0.12f, -0.4f}, {0.12f, -0.26f}}, false);
            poly({{-0.28f, -0.26f}, {-0.22f, 0.4f}, {0.22f, 0.4f}, {0.28f, -0.26f}}, false);
            break;
        case IC_LAYERS:
            poly({{0.f, -0.4f}, {0.42f, -0.18f}, {0.f, 0.04f}, {-0.42f, -0.18f}}, true);
            poly({{-0.42f, 0.04f}, {0.f, 0.26f}, {0.42f, 0.04f}}, false);
            poly({{-0.42f, 0.22f}, {0.f, 0.44f}, {0.42f, 0.22f}}, false);
            break;
        case IC_PERCENT:
            line(0.3f, -0.34f, -0.3f, 0.34f);
            circle(-0.22f, -0.22f, 0.1f);
            circle(0.22f, 0.22f, 0.1f);
            break;
    }
}

// The Kor mark: an ember (flame drop) holding a three-node network.
void draw_logo(ImDrawList* dl, ImVec2 p, float sz) {
    auto P = [&](float x, float y) { return ImVec2(p.x + x / 256.f * sz, p.y + y / 256.f * sz); };
    const ImU32 dark = IM_COL32(23, 18, 15, 255);
    dl->AddRectFilled(p, ImVec2(p.x + sz, p.y + sz), dark, sz * 0.24f);
    dl->AddRect(p, ImVec2(p.x + sz, p.y + sz), col(C::accent, 0.25f), sz * 0.24f, 0, std::max(1.f, sz * 0.02f));
    dl->PathLineTo(P(128, 26));
    dl->PathBezierCubicCurveTo(P(150, 70), P(202, 100), P(202, 154));
    dl->PathArcTo(P(128, 154), 74.f / 256.f * sz, 0.f, kPi, 28);
    dl->PathBezierCubicCurveTo(P(54, 100), P(106, 70), P(128, 26));
    dl->PathFillConcave(col(C::accent));
    ImVec2 n0 = P(128, 112), n1 = P(94, 172), n2 = P(162, 172);
    float lw = sz * 0.035f;
    dl->AddLine(n0, n1, dark, lw);
    dl->AddLine(n0, n2, dark, lw);
    dl->AddLine(n1, n2, dark, lw);
    for (ImVec2 n : {n0, n1, n2}) {
        dl->AddCircleFilled(n, sz * 0.062f, dark, 20);
        dl->AddCircleFilled(n, sz * 0.026f, col(ImVec4(1.f, 0.76f, 0.29f, 1.f)), 12);
    }
}

// ------------------------------------------------------------------- text

void text(const char* s, const ImVec4& c) {
    ImGui::PushStyleColor(ImGuiCol_Text, c);
    ImGui::TextUnformatted(s);
    ImGui::PopStyleColor();
}

void text_wrapped(const char* s, const ImVec4& c, ImFont* f) {
    FontScope fs(f);
    ImGui::PushStyleColor(ImGuiCol_Text, c);
    ImGui::PushTextWrapPos(0.f);
    ImGui::TextUnformatted(s);
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
}

void section_title(const char* s) {
    FontScope f(F.h2);
    ImGui::TextUnformatted(s);
}

void tooltip(const char* s) {
    if (!s || !*s || !ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort | ImGuiHoveredFlags_AllowWhenDisabled)) return;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(px(12), px(9)));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, C::surface3);
    ImGui::PushStyleColor(ImGuiCol_Border, C::border_hi);
    if (ImGui::BeginTooltip()) {
        ImGui::PushTextWrapPos(px(380));
        ImGui::TextUnformatted(s);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar();
}

void gap(float h) { ImGui::Dummy(ImVec2(0, px(h))); }

// ----------------------------------------------------------------- layout

void header(const char* title, const char* subtitle, float actions_w) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    float w = ImGui::GetContentRegionAvail().x;
    g_hdr_start = p;
    float text_w = w - (actions_w > 0 ? actions_w + px(28) : 0);
    ImGui::BeginGroup();
    {
        FontScope f(F.h1);
        ImGui::TextUnformatted(title);
    }
    if (subtitle && *subtitle) {
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + std::max(px(220), std::min(text_w, px(860))));
        ImGui::PushStyleColor(ImGuiCol_Text, C::muted);
        ImGui::TextUnformatted(subtitle);
        ImGui::PopStyleColor();
        ImGui::PopTextWrapPos();
    }
    ImGui::EndGroup();
    g_hdr_bottom = ImGui::GetItemRectMax().y;
    if (actions_w > 0) {
        float h1 = F.h1 ? F.h1->FontSize : ImGui::GetFontSize();
        ImGui::SetCursorScreenPos(ImVec2(p.x + w - actions_w, p.y + std::max(0.f, (h1 - ImGui::GetFrameHeight()) * 0.5f)));
    }
}

void header_end() {
    float bottom = std::max(g_hdr_bottom, ImGui::GetItemRectMax().y);
    ImGui::SetCursorScreenPos(ImVec2(g_hdr_start.x, bottom + px(24) - ImGui::GetStyle().ItemSpacing.y));
    ImGui::Dummy(ImVec2(0, 0));
}

int fit_columns(float min_w, int max_cols, float g) {
    float avail = ImGui::GetContentRegionAvail().x;
    int n = (int)((avail + g) / (min_w + g));
    return std::max(1, std::min(max_cols, n));
}

float column_width(int cols, float g) {
    float avail = ImGui::GetContentRegionAvail().x;
    return std::floor((avail - g * (cols - 1)) / cols);
}

float card_padding_y() { return px(18); }

static bool g_relayout = false;

bool take_relayout() {
    bool r = g_relayout;
    g_relayout = false;
    return r;
}

void begin_card(const char* id, const char* title, float min_h, float width, const char* subtitle) {
    // ImGui hides an auto-sized child on its first frame while it measures it.
    ImGuiStorage* seen = ImGui::GetStateStorage();
    ImGuiID key = ImGui::GetID(id) ^ 0x63617264u;
    if (!seen->GetBool(key)) {
        seen->SetBool(key, true);
        g_relayout = true;
    }
    ImGui::PushStyleColor(ImGuiCol_ChildBg, C::surface);
    ImGui::PushStyleColor(ImGuiCol_Border, C::border);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(px(20), card_padding_y()));
    if (min_h > 0) ImGui::SetNextWindowSizeConstraints(ImVec2(0, min_h), ImVec2(FLT_MAX, FLT_MAX));
    ImGui::BeginChild(id, ImVec2(width, 0),
                      ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor(2);
    if (!title) return;
    {
        FontScope f(F.bold);
        ImGui::TextUnformatted(title);
    }
    if (subtitle && *subtitle) text_wrapped(subtitle, C::muted, F.small);
    ImVec2 wp = ImGui::GetWindowPos();
    float y = ImGui::GetCursorScreenPos().y + px(2);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->PushClipRect(wp, ImVec2(wp.x + ImGui::GetWindowWidth(), wp.y + ImGui::GetWindowHeight()), false);
    dl->AddLine(ImVec2(wp.x + 1, y), ImVec2(wp.x + ImGui::GetWindowWidth() - 1, y), col(C::border));
    dl->PopClipRect();
    ImGui::Dummy(ImVec2(0, px(8)));
}

void end_card(float* natural_h) {
    if (natural_h) {
        float h = ImGui::GetCursorPosY() - ImGui::GetStyle().ItemSpacing.y + card_padding_y();
        *natural_h = std::max(*natural_h, h);
    }
    ImGui::EndChild();
}

bool begin_collapsible_card(const char* id, const char* title, bool* open, const char* summary) {
    begin_card(id);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float w = ImGui::GetContentRegionAvail().x;
    float h = std::max(F.bold ? F.bold->FontSize : ImGui::GetFontSize(), px(22));
    if (ImGui::InvisibleButton("##fold", ImVec2(w, h))) *open = !*open;
    bool hov = ImGui::IsItemHovered();
    if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    float a = anim(ImGui::GetItemID(), *open, 18.f);
    float cy = p.y + h * 0.5f;
    icon(dl, a > 0.5f ? IC_CHEVRON_DOWN : IC_CHEVRON_RIGHT, ImVec2(p.x + px(7), cy), px(14), col(hov ? C::text : C::muted));
    ImFont* bf = F.bold ? F.bold : ImGui::GetFont();
    ImVec2 ts = text_size(bf, title);
    dl->AddText(bf, bf->FontSize, ImVec2(p.x + px(22), cy - ts.y * 0.5f), col(C::text), title);
    if (summary && *summary) {
        ImFont* sf = F.small ? F.small : ImGui::GetFont();
        float x0 = p.x + px(22) + ts.x + px(16);
        float sw = text_size(sf, summary).x;
        float avail = p.x + w - x0;
        if (avail > px(60))
            text_ellipsis(dl, sf, ImVec2(std::max(x0, p.x + w - sw), cy - sf->FontSize * 0.5f), avail, summary, col(C::muted));
    }
    if (*open) {
        ImVec2 wp = ImGui::GetWindowPos();
        float y = ImGui::GetCursorScreenPos().y + px(4);
        dl->PushClipRect(wp, ImVec2(wp.x + ImGui::GetWindowWidth(), wp.y + ImGui::GetWindowHeight()), false);
        dl->AddLine(ImVec2(wp.x + 1, y), ImVec2(wp.x + ImGui::GetWindowWidth() - 1, y), col(C::border));
        dl->PopClipRect();
        ImGui::Dummy(ImVec2(0, px(10)));
    }
    return *open;
}

// --------------------------------------------------------------- controls

float button_width(const char* label, Icon ic) {
    ImFont* f = F.bold ? F.bold : ImGui::GetFont();
    float tw = text_size(f, label, label_end(label)).x;
    float iw = ic != IC_NONE ? px(16) + (tw > 0 ? px(8) : 0) : 0;
    if (tw == 0 && ic != IC_NONE) return ImGui::GetFrameHeight();
    return tw + iw + px(16) * 2;
}

bool button(const char* label, ButtonKind kind, Icon ic, float width, float height) {
    ImFont* f = F.bold ? F.bold : ImGui::GetFont();
    const char* le = label_end(label);
    ImVec2 ts = text_size(f, label, le);
    float h = height > 0 ? height : ImGui::GetFrameHeight();
    float w = width > 0 ? width : button_width(label, ic);
    ImVec2 p = ImGui::GetCursorScreenPos();
    bool clicked = ImGui::InvisibleButton(label, ImVec2(w, h));
    bool hov = ImGui::IsItemHovered(), act = ImGui::IsItemActive();
    float t = anim(ImGui::GetItemID(), hov, 20.f);
    ImVec4 bg, fg, br(0, 0, 0, 0);
    switch (kind) {
        case BTN_PRIMARY:
            bg = act ? C::accent_lo : mix(C::accent, C::accent_hi, t * 0.6f);
            fg = C::on_accent;
            break;
        case BTN_SECONDARY:
            bg = act ? C::border_hi : mix(C::surface2, C::surface3, t);
            fg = C::text;
            br = mix(C::border_hi, mix(C::border_hi, C::muted, 0.3f), t);
            break;
        case BTN_GHOST:
            bg = with_alpha(C::surface3, act ? 1.f : t);
            fg = mix(C::muted, C::text, t);
            break;
        case BTN_DANGER:
            bg = with_alpha(C::bad, act ? 0.28f : 0.10f + 0.10f * t);
            fg = mix(C::bad, ImVec4(1, 0.75f, 0.75f, 1), t * 0.5f);
            br = with_alpha(C::bad, 0.35f);
            break;
    }
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float r = px(8);
    if (bg.w > 0) dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(bg), r);
    if (br.w > 0) dl->AddRect(p, ImVec2(p.x + w, p.y + h), col(br), r, 0, 1.f);
    float iw = ic != IC_NONE ? px(16) + (ts.x > 0 ? px(8) : 0) : 0;
    float x = p.x + std::floor((w - iw - ts.x) * 0.5f);
    float cy = p.y + h * 0.5f;
    if (ic != IC_NONE) icon(dl, ic, ImVec2(x + px(8), cy), px(15), col(fg));
    if (ts.x > 0) dl->AddText(f, f->FontSize, ImVec2(x + iw, std::floor(cy - ts.y * 0.5f)), col(fg), label, le);
    if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    return clicked;
}

bool segmented(const char* id, int* value, const char* const* labels, int n) {
    ImGui::PushID(id);
    ImFont* f = F.bold ? F.bold : ImGui::GetFont();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float h = ImGui::GetFrameHeight(), pad = px(3);
    std::vector<float> ws(n);
    float total = pad * 2;
    for (int i = 0; i < n; ++i) total += ws[i] = text_size(f, labels[i]).x + px(18) * 2;
    dl->AddRectFilled(p, ImVec2(p.x + total, p.y + h), col(C::inset), px(9));
    dl->AddRect(p, ImVec2(p.x + total, p.y + h), col(C::border), px(9));
    bool changed = false;
    float x = p.x + pad;
    for (int i = 0; i < n; ++i) {
        ImGui::PushID(i);
        ImGui::SetCursorScreenPos(ImVec2(x, p.y + pad));
        if (ImGui::InvisibleButton("##seg", ImVec2(ws[i], h - pad * 2)) && *value != i) { *value = i; changed = true; }
        bool hov = ImGui::IsItemHovered(), sel = *value == i;
        if (hov && !sel) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        if (sel) {
            dl->AddRectFilled(ImVec2(x, p.y + pad), ImVec2(x + ws[i], p.y + h - pad), col(C::surface3), px(7));
            dl->AddRect(ImVec2(x, p.y + pad), ImVec2(x + ws[i], p.y + h - pad), col(C::border_hi), px(7));
        }
        ImVec2 ts = text_size(f, labels[i]);
        dl->AddText(f, f->FontSize, ImVec2(x + (ws[i] - ts.x) * 0.5f, p.y + (h - ts.y) * 0.5f),
                    col(sel ? C::text : hov ? mix(C::muted, C::text, 0.6f) : C::muted), labels[i]);
        x += ws[i];
        ImGui::PopID();
    }
    ImGui::SetCursorScreenPos(p);
    ImGui::Dummy(ImVec2(total, h));
    ImGui::PopID();
    return changed;
}

bool tabs(const char* id, int* value, const char* const* labels, int n) {
    ImGui::PushID(id);
    ImFont* f = F.bold ? F.bold : ImGui::GetFont();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float w = ImGui::GetContentRegionAvail().x;
    float h = f->FontSize + px(16);
    dl->AddLine(ImVec2(p.x, p.y + h - 0.5f), ImVec2(p.x + w, p.y + h - 0.5f), col(C::border));
    bool changed = false;
    float x = p.x;
    for (int i = 0; i < n; ++i) {
        ImGui::PushID(i);
        ImVec2 ts = text_size(f, labels[i]);
        float tw = ts.x + px(8);
        ImGui::SetCursorScreenPos(ImVec2(x, p.y));
        if (ImGui::InvisibleButton("##tab", ImVec2(tw, h)) && *value != i) { *value = i; changed = true; }
        bool hov = ImGui::IsItemHovered(), sel = *value == i;
        if (hov && !sel) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        float a = anim(ImGui::GetItemID(), sel, 16.f);
        dl->AddText(f, f->FontSize, ImVec2(x + px(4), p.y + px(6)), col(sel ? C::text : hov ? mix(C::muted, C::text, 0.6f) : C::muted), labels[i]);
        if (a > 0.01f) {
            float cx = x + tw * 0.5f, hw = tw * 0.5f * a;
            dl->AddRectFilled(ImVec2(cx - hw, p.y + h - px(2)), ImVec2(cx + hw, p.y + h), col(C::accent), px(1));
        }
        x += tw + px(22);
        ImGui::PopID();
    }
    ImGui::SetCursorScreenPos(p);
    ImGui::Dummy(ImVec2(w, h));
    ImGui::PopID();
    return changed;
}

bool nav_item(const char* label, Icon ic, bool selected, int status) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float w = ImGui::GetContentRegionAvail().x, h = px(38);
    bool clicked = ImGui::InvisibleButton(label, ImVec2(w, h));
    bool hov = ImGui::IsItemHovered();
    if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    float t = anim(ImGui::GetItemID(), hov && !selected, 16.f);
    if (selected) {
        dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(C::accent, 0.12f), px(8));
        dl->AddRectFilled(ImVec2(p.x, p.y + px(10)), ImVec2(p.x + px(3), p.y + h - px(10)), col(C::accent), px(2));
    } else if (t > 0.01f) {
        dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(C::surface2, t), px(8));
    }
    float cy = p.y + h * 0.5f;
    icon(dl, ic, ImVec2(p.x + px(22), cy), px(17), col(selected ? C::accent : mix(C::muted, C::text, t)));
    ImFont* f = selected && F.bold ? F.bold : ImGui::GetFont();
    float right = status ? px(26) : px(8);
    text_ellipsis(dl, f, ImVec2(p.x + px(44), cy - f->FontSize * 0.5f), w - px(44) - right, label,
                  col(selected ? C::text : mix(mix(C::muted, C::text, 0.45f), C::text, t)));
    if (status == 1) {
        ImVec2 c(p.x + w - px(16), cy);
        dl->AddCircleFilled(c, px(7), col(C::good, 0.16f));
        icon(dl, IC_CHECK, c, px(10), col(C::good));
    } else if (status == 2) {
        float pulse = 0.5f + 0.5f * std::sin((float)ImGui::GetTime() * 5.f);
        ImVec2 c(p.x + w - px(16), cy);
        dl->AddCircleFilled(c, px(3.5f) + px(3.5f) * pulse, col(C::accent, 0.25f * (1.f - pulse)));
        dl->AddCircleFilled(c, px(3.5f), col(C::accent));
    }
    return clicked;
}

float badge_width(const char* text, bool dot) {
    ImFont* f = F.small_bold ? F.small_bold : ImGui::GetFont();
    return text_size(f, text).x + px(10) * 2 + (dot ? px(12) : 0);
}

void badge(const char* text, const ImVec4& c, bool dot, bool pulse) {
    ImFont* f = F.small_bold ? F.small_bold : ImGui::GetFont();
    ImVec2 ts = text_size(f, text);
    float h = ts.y + px(8), w = badge_width(text, dot);
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(w, h));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(c, 0.13f), h * 0.5f);
    dl->AddRect(p, ImVec2(p.x + w, p.y + h), col(c, 0.30f), h * 0.5f);
    float x = p.x + px(10), cy = p.y + h * 0.5f;
    if (dot) {
        ImVec2 dc(x + px(3), cy);
        if (pulse) {
            float k = 0.5f + 0.5f * std::sin((float)ImGui::GetTime() * 5.f);
            dl->AddCircleFilled(dc, px(3) + px(3) * k, col(c, 0.3f * (1.f - k)));
        }
        dl->AddCircleFilled(dc, px(3), col(c));
        x += px(12);
    }
    dl->AddText(f, f->FontSize, ImVec2(x, p.y + px(4)), col(c), text);
}

void progress_bar(float frac, float height, const ImVec4& c) {
    frac = std::max(0.f, std::min(1.f, frac));
    ImVec2 p = ImGui::GetCursorScreenPos();
    float w = ImGui::GetContentRegionAvail().x;
    ImGui::Dummy(ImVec2(w, height));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float r = height * 0.5f;
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + height), col(C::surface3), r);
    if (frac > 0) {
        float fw = std::max(height, w * frac);
        dl->AddRectFilled(p, ImVec2(p.x + fw, p.y + height), col(c), r);
        dl->AddRectFilled(ImVec2(p.x + r * 0.5f, p.y), ImVec2(p.x + fw - r * 0.5f, p.y + height * 0.45f), col(ImVec4(1, 1, 1, 1), 0.10f), r);
    }
}

// t is the knob position in [0, 1]; steps > 0 snaps it to that many intervals.
static bool slider_core(const char* id, float* t, int steps, const char* value, const char* widest) {
    float w = ImGui::CalcItemWidth();
    float h = ImGui::GetFrameHeight();
    ImFont* f = ImGui::GetFont();
    float vw = std::max(px(52), text_size(f, widest).x + px(22));
    float tw = std::max(px(40), w - vw - px(12));
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::PushID(id);
    ImGui::InvisibleButton("##track", ImVec2(tw, h));
    bool active = ImGui::IsItemActive(), hovered = ImGui::IsItemHovered();
    if (hovered || active) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    float r = px(7.5f);
    float x0 = p.x + r, x1 = p.x + tw - r, cy = p.y + h * 0.5f;
    bool changed = false;
    if (active && x1 > x0) {
        float nt = std::max(0.f, std::min(1.f, (ImGui::GetIO().MousePos.x - x0) / (x1 - x0)));
        if (steps > 0) nt = std::round(nt * steps) / steps;
        if (nt != *t) {
            *t = nt;
            changed = true;
        }
    }
    float k = anim(ImGui::GetItemID(), hovered || active, 14.f);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float kx = x0 + (x1 - x0) * std::max(0.f, std::min(1.f, *t));
    float th = px(4);
    dl->AddRectFilled(ImVec2(x0, cy - th * 0.5f), ImVec2(x1, cy + th * 0.5f), col(C::surface3), th);
    dl->AddRectFilled(ImVec2(x0, cy - th * 0.5f), ImVec2(std::max(kx, x0 + th), cy + th * 0.5f), col(active ? C::accent_hi : C::accent), th);
    if (k > 0.01f) dl->AddCircleFilled(ImVec2(kx, cy), r + px(5) * k, col(C::accent, 0.18f * k), 32);
    dl->AddCircleFilled(ImVec2(kx, cy + px(1)), r + px(1), col(ImVec4(0, 0, 0, 1), 0.35f), 32);
    dl->AddCircleFilled(ImVec2(kx, cy), r, col(C::text), 32);
    dl->AddCircleFilled(ImVec2(kx, cy), r * 0.38f, col(C::accent), 16);

    ImGui::SameLine(0, px(12));
    ImVec2 b = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(vw, h));
    float fr = ImGui::GetStyle().FrameRounding;
    dl->AddRectFilled(b, ImVec2(b.x + vw, b.y + h), col(C::inset), fr);
    dl->AddRect(b, ImVec2(b.x + vw, b.y + h), col(active ? C::accent_lo : C::border), fr);
    ImVec2 ts = text_size(f, value);
    dl->AddText(f, f->FontSize, ImVec2(b.x + (vw - ts.x) * 0.5f, b.y + (h - ts.y) * 0.5f), col(C::text), value);
    ImGui::PopID();
    return changed;
}

bool slider_int(const char* id, int* v, int lo, int hi, const char* fmt) {
    if (hi <= lo) hi = lo + 1;
    *v = std::max(lo, std::min(hi, *v));
    float t = float(*v - lo) / float(hi - lo);
    char value[64], widest[64];
    snprintf(value, sizeof value, fmt, *v);
    snprintf(widest, sizeof widest, fmt, std::abs(lo) > std::abs(hi) ? lo : hi);
    if (!slider_core(id, &t, hi - lo, value, widest)) return false;
    int nv = lo + (int)std::lround(t * (hi - lo));
    if (nv == *v) return false;
    *v = nv;
    return true;
}

bool slider_float(const char* id, float* v, float lo, float hi, const char* fmt) {
    if (hi <= lo) hi = lo + 1;
    *v = std::max(lo, std::min(hi, *v));
    float t = (*v - lo) / (hi - lo);
    char value[64], widest[64];
    snprintf(value, sizeof value, fmt, *v);
    snprintf(widest, sizeof widest, fmt, std::fabs(lo) > std::fabs(hi) ? lo : hi);
    if (!slider_core(id, &t, 0, value, widest)) return false;
    *v = lo + t * (hi - lo);
    return true;
}

bool begin_combo(const char* id, const char* preview) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    float w = ImGui::CalcItemWidth(), h = ImGui::GetFrameHeight();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImGui::SetNextItemWidth(w);
    std::string shown = preview ? preview : "";
    float room = w - h - ImGui::GetStyle().FramePadding.x;
    if (ImGui::CalcTextSize(shown.c_str()).x > room) {
        while (!shown.empty() && ImGui::CalcTextSize((shown + "...").c_str()).x > room) {
            size_t cut = shown.size() - 1;
            while (cut > 0 && (shown[cut] & 0xC0) == 0x80) --cut;
            shown.erase(cut);
        }
        shown += "...";
    }
    bool open = ImGui::BeginCombo(id, shown.c_str(), ImGuiComboFlags_NoArrowButton);
    icon(dl, IC_CHEVRON_DOWN, ImVec2(p.x + w - h * 0.5f, p.y + h * 0.5f), px(14), col(open ? C::text : C::muted));
    return open;
}

bool combo(const char* id, int* current, const char* const* items, int n) {
    bool changed = false;
    if (begin_combo(id, *current >= 0 && *current < n ? items[*current] : "")) {
        for (int i = 0; i < n; ++i)
            if (ImGui::Selectable(items[i], i == *current) && i != *current) {
                *current = i;
                changed = true;
            }
        ImGui::EndCombo();
    }
    return changed;
}

bool checkbox(const char* label, bool* v) {
    float h = ImGui::GetFrameHeight();
    float bs = px(18);
    const char* end = strstr(label, "##");
    if (!end) end = label + strlen(label);
    float lw = end > label ? ImGui::CalcTextSize(label, end).x + px(10) : 0.f;
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton(label, ImVec2(bs + lw, h));
    bool clicked = ImGui::IsItemClicked();
    if (clicked) *v = !*v;
    bool hovered = ImGui::IsItemHovered();
    float k = anim(ImGui::GetItemID(), *v, 16.f);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 b0(p.x, p.y + (h - bs) * 0.5f), b1(b0.x + bs, b0.y + bs);
    float rr = px(5);
    dl->AddRectFilled(b0, b1, col(C::inset), rr);
    dl->AddRect(b0, b1, col(hovered ? C::muted : C::border_hi), rr, 0, px(1.2f));
    if (k > 0.01f) {
        dl->AddRectFilled(b0, b1, col(C::accent, k), rr);
        icon(dl, IC_CHECK, ImVec2(b0.x + bs * 0.5f, b0.y + bs * 0.5f), bs * 0.8f, col(C::on_accent, k));
    }
    if (lw > 0) {
        float ty = p.y + (h - ImGui::GetFontSize()) * 0.5f;
        dl->AddText(ImVec2(b1.x + px(10), ty), col(C::text), label, end);
    }
    return clicked;
}

void ring(float frac, float radius, float th, const ImVec4& c, const char* center) {
    frac = std::max(0.f, std::min(1.f, frac));
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(radius * 2, radius * 2));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 ctr(p.x + radius, p.y + radius);
    float r = radius - th * 0.5f;
    dl->AddCircle(ctr, r, col(C::surface3), 64, th);
    if (frac > 0.001f) {
        float a0 = -kPi * 0.5f;
        dl->PathArcTo(ctr, r, a0, a0 + 2 * kPi * frac, std::max(6, (int)(64 * frac)));
        dl->PathStroke(col(c), 0, th);
    }
    if (center) {
        ImFont* f = F.h2 ? F.h2 : ImGui::GetFont();
        ImVec2 ts = text_size(f, center);
        dl->AddText(f, f->FontSize, ImVec2(ctr.x - ts.x * 0.5f, ctr.y - ts.y * 0.5f), col(C::text), center);
    }
}

void field_label(const char* label, float label_w, const char* help) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    float h = ImGui::GetFrameHeight();
    ImGui::Dummy(ImVec2(label_w, h));
    bool hov = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImFont* f = ImGui::GetFont();
    float help_w = help ? px(20) : 0;
    bool cut = text_ellipsis(dl, f, ImVec2(p.x, p.y + (h - f->FontSize) * 0.5f), label_w - px(8) - help_w, label, col(C::muted));
    if (help) {
        float tw = std::min(text_size(f, label).x, label_w - px(8) - help_w);
        icon(dl, IC_INFO, ImVec2(p.x + tw + px(12), p.y + h * 0.5f), px(14), col(hov ? C::text : C::subtle));
    }
    if (hov && (cut || help)) {
        std::string tip = cut ? std::string(label) : std::string();
        if (help) tip += (tip.empty() ? "" : "\n\n") + std::string(help);
        tooltip(tip.c_str());
    }
    ImGui::SameLine(0, px(12));
    ImGui::SetNextItemWidth(-FLT_MIN);
}

void banner(const char* text, const ImVec4& c, Icon ic) {
    float w = ImGui::GetContentRegionAvail().x;
    float padx = px(14), pady = px(11), iw = px(28);
    float wrap = w - padx * 2 - iw;
    ImVec2 ts = text_size(ImGui::GetFont(), text, nullptr, wrap);
    float h = std::max(ts.y, px(18)) + pady * 2;
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(c, 0.09f), px(10));
    dl->AddRect(p, ImVec2(p.x + w, p.y + h), col(c, 0.28f), px(10));
    icon(dl, ic, ImVec2(p.x + padx + px(9), p.y + pady + ImGui::GetFontSize() * 0.5f), px(16), col(c));
    dl->AddText(ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(p.x + padx + iw, p.y + pady), col(C::text), text, nullptr, wrap);
    ImGui::Dummy(ImVec2(w, h));
}

int empty_state(Icon ic, const char* title, const char* body, const char* b1, const char* b2) {
    int r = 0;
    begin_card("##empty", nullptr, px(280));
    float w = ImGui::GetContentRegionAvail().x;
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    gap(28);
    ImVec2 c(p.x + w * 0.5f, ImGui::GetCursorScreenPos().y + px(28));
    dl->AddCircleFilled(c, px(28), col(C::accent, 0.12f));
    dl->AddCircle(c, px(28), col(C::accent, 0.25f), 0, 1.f);
    icon(dl, ic, c, px(24), col(C::accent));
    ImGui::Dummy(ImVec2(0, px(64)));
    auto centered = [&](const char* s, ImFont* f, const ImVec4& color) {
        float bw = std::min(w, px(480));
        ImVec2 ts = text_size(f, s);
        FontScope fs(f);
        ImGui::PushStyleColor(ImGuiCol_Text, color);
        if (ts.x <= bw) {
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (w - ts.x) * 0.5f);
            ImGui::TextUnformatted(s);
        } else {
            float x0 = ImGui::GetCursorPosX() + (w - bw) * 0.5f;
            ImGui::SetCursorPosX(x0);
            ImGui::PushTextWrapPos(x0 + bw);
            ImGui::TextUnformatted(s);
            ImGui::PopTextWrapPos();
        }
        ImGui::PopStyleColor();
    };
    if (title) centered(title, F.h2 ? F.h2 : ImGui::GetFont(), C::text);
    if (body) centered(body, ImGui::GetFont(), C::muted);
    if (b1) {
        gap(6);
        float w1 = button_width(b1), w2 = b2 ? button_width(b2) : 0, g = px(10);
        float total = w1 + (b2 ? g + w2 : 0);
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (w - total) * 0.5f);
        if (button(b1, BTN_PRIMARY)) r = 1;
        if (b2) {
            ImGui::SameLine(0, g);
            if (button(b2, BTN_SECONDARY)) r = 2;
        }
    }
    gap(20);
    end_card();
    return r;
}

float stat_tile_height(bool big, bool sub) {
    float small = F.small ? F.small->FontSize : ImGui::GetFontSize();
    float value = big ? (F.stat ? F.stat->FontSize : ImGui::GetFontSize()) : (F.bold ? F.bold->FontSize : ImGui::GetFontSize());
    return std::ceil(px(16) * 2 + small + px(8) + value + (sub ? px(4) + small : 0));
}

void stat_tile(const Stat& s, float w, float h, bool big) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(w, h));
    bool hov = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(C::surface), px(12));
    dl->AddRect(p, ImVec2(p.x + w, p.y + h), col(C::border), px(12));
    float pad = px(16);
    ImFont* sf = F.small ? F.small : ImGui::GetFont();
    ImFont* vf = big ? (F.stat ? F.stat : ImGui::GetFont()) : (F.bold ? F.bold : ImGui::GetFont());
    float icon_w = s.ic != IC_NONE ? px(22) : 0;
    if (s.ic != IC_NONE) icon(dl, s.ic, ImVec2(p.x + w - pad - px(8), p.y + pad + sf->FontSize * 0.5f), px(16), col(C::subtle));
    bool cut = text_ellipsis(dl, sf, ImVec2(p.x + pad, p.y + pad), w - pad * 2 - icon_w, s.label.c_str(), col(C::muted));
    float y = p.y + pad + sf->FontSize + px(8);
    cut |= text_ellipsis(dl, vf, ImVec2(p.x + pad, y), w - pad * 2, s.value.c_str(), col(s.color));
    if (!s.sub.empty())
        cut |= text_ellipsis(dl, sf, ImVec2(p.x + pad, y + vf->FontSize + px(4)), w - pad * 2, s.sub.c_str(), col(C::muted));
    if (hov && cut) {
        std::string tip = s.label + "\n" + s.value + (s.sub.empty() ? "" : "\n" + s.sub);
        tooltip(tip.c_str());
    }
}

void stat_grid(const std::vector<Stat>& stats, float min_w, bool big) {
    if (stats.empty()) return;
    float g = px(12);
    int cols = fit_columns(min_w, (int)stats.size(), g);
    float w = column_width(cols, g);
    bool any_sub = false;
    for (auto& s : stats) any_sub |= !s.sub.empty();
    float h = stat_tile_height(big, any_sub);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(g, g));
    for (size_t i = 0; i < stats.size(); ++i) {
        if (i % cols) ImGui::SameLine(0, g);
        ImGui::PushID((int)i);
        stat_tile(stats[i], w, h, big);
        ImGui::PopID();
    }
    ImGui::PopStyleVar();
}

}  // namespace kit
