#include "ui.h"

#include "i18n.h"
#include "platform.h"

#include "../core/kernels.h"
#include "../core/sysinfo.h"

#include "imgui.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace kor;

namespace app {
namespace {

// ------------------------------------------------------------------ state

struct State {
    float sc = 1.f;
    ImFont* title_font = nullptr;
    int page = PAGE_HOME;

    SysInfo sys;
    Dataset data;
    std::shared_ptr<Model> model;
    Trainer trainer;
    ModelSpec spec;
    TrainConfig cfg;
    AutoPlan plan;
    bool has_plan = false;

    char data_path[2048] = "";
    int data_kind = 0;  // 0 text, 1 table
    std::string data_msg;
    bool data_ok = true;

    int add_type = L_DENSE;
    int lora_rank = 8;

    char prompt[4096] = "";
    float temperature = 0.8f;
    int gen_len = 300;
    std::vector<int> gen_ids;
    int gen_left = 0;
    std::string gen_out;
    Rng gen_rng{2024};
    std::vector<float> feat_in;
    std::vector<float> pred;

    char model_path[2048] = "";
    bool half = false;
    std::string file_msg;
    bool file_ok = true;
};
State A;

const ImVec4 kEmber(1.00f, 0.48f, 0.10f, 1.f);
const ImVec4 kEmberHi(1.00f, 0.76f, 0.29f, 1.f);
const ImVec4 kMuted(0.64f, 0.60f, 0.57f, 1.f);
const ImVec4 kGood(0.45f, 0.85f, 0.55f, 1.f);
const ImVec4 kBad(1.00f, 0.42f, 0.36f, 1.f);
const ImVec4 kBlue(0.42f, 0.72f, 1.00f, 1.f);

bool busy() { return A.trainer.running(); }

// ---------------------------------------------------------------- helpers

std::string fmt_count(double n) {
    char b[32];
    if (n >= 1e9) std::snprintf(b, sizeof b, "%.2fB", n / 1e9);
    else if (n >= 1e6) std::snprintf(b, sizeof b, "%.2fM", n / 1e6);
    else if (n >= 1e4) std::snprintf(b, sizeof b, "%.1fK", n / 1e3);
    else std::snprintf(b, sizeof b, "%.0f", n);
    return b;
}

std::string fmt_time(double s) {
    char b[32];
    int t = (int)(s + 0.5);
    if (t >= 3600) std::snprintf(b, sizeof b, "%d:%02d:%02d", t / 3600, t / 60 % 60, t % 60);
    else std::snprintf(b, sizeof b, "%d:%02d", t / 60, t % 60);
    return b;
}

std::string fmt_mb(double mb) {
    char b[32];
    if (mb >= 1024) std::snprintf(b, sizeof b, "%.1f GB", mb / 1024);
    else std::snprintf(b, sizeof b, "%.0f MB", mb < 1 ? 1.0 : mb);
    return b;
}

void Title(const char* s) {
    if (A.title_font) ImGui::PushFont(A.title_font);
    ImGui::TextUnformatted(s);
    if (A.title_font) ImGui::PopFont();
    ImGui::Dummy(ImVec2(0, 4 * A.sc));
}

void Muted(const char* s) {
    ImGui::PushStyleColor(ImGuiCol_Text, kMuted);
    ImGui::TextWrapped("%s", s);
    ImGui::PopStyleColor();
}

void Row(const char* label, const char* value, const ImVec4* color = nullptr) {
    ImGui::TextColored(kMuted, "%s", label);
    ImGui::SameLine(230 * A.sc);
    if (color) ImGui::TextColored(*color, "%s", value);
    else ImGui::TextWrapped("%s", value);
}

bool PrimaryButton(const char* label, ImVec2 size = ImVec2(0, 0)) {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.90f, 0.40f, 0.06f, 1));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, kEmber);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.78f, 0.33f, 0.04f, 1));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1));
    bool r = ImGui::Button(label, size);
    ImGui::PopStyleColor(4);
    return r;
}

bool BeginCard(const char* id, float height = 0) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.115f, 0.100f, 0.092f, 1));
    ImGuiChildFlags flags = ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding;
    if (height == 0) flags |= ImGuiChildFlags_AutoResizeY;
    bool open = ImGui::BeginChild(id, ImVec2(0, height), flags);
    ImGui::PopStyleColor();
    return open;
}
void EndCard() {
    ImGui::EndChild();
    ImGui::Dummy(ImVec2(0, 6 * A.sc));
}

const char* layer_name(int type) {
    switch (type) {
        case L_DENSE: return tr(S::LDense);
        case L_RELU: return "ReLU";
        case L_GELU: return "GELU";
        case L_TANH: return "Tanh";
        case L_SIGMOID: return "Sigmoid";
        case L_DROPOUT: return "Dropout";
        case L_LAYERNORM: return "LayerNorm";
    }
    return "?";
}

const char* tier_name(int t) {
    static const char* const n[] = {"nano", "micro", "small", "base", "large"};
    return n[t < 0 ? 0 : t > 4 ? 4 : t];
}

// The Kor mark: an ember (flame drop) holding a three-node network.
void draw_logo(ImDrawList* dl, ImVec2 p, float sz) {
    auto P = [&](float x, float y) { return ImVec2(p.x + x / 256.f * sz, p.y + y / 256.f * sz); };
    const ImU32 dark = IM_COL32(23, 18, 15, 255);
    dl->AddRectFilled(p, ImVec2(p.x + sz, p.y + sz), dark, sz * 0.22f);
    dl->PathLineTo(P(128, 26));
    dl->PathBezierCubicCurveTo(P(150, 70), P(202, 100), P(202, 154));
    dl->PathArcTo(P(128, 154), 74.f / 256.f * sz, 0.f, 3.14159265f, 28);
    dl->PathBezierCubicCurveTo(P(54, 100), P(106, 70), P(128, 26));
    dl->PathFillConcave(ImGui::GetColorU32(kEmber));
    ImVec2 n0 = P(128, 112), n1 = P(94, 172), n2 = P(162, 172);
    float lw = sz * 0.035f;
    dl->AddLine(n0, n1, dark, lw);
    dl->AddLine(n0, n2, dark, lw);
    dl->AddLine(n1, n2, dark, lw);
    for (ImVec2 n : {n0, n1, n2}) {
        dl->AddCircleFilled(n, sz * 0.062f, dark, 20);
        dl->AddCircleFilled(n, sz * 0.026f, ImGui::GetColorU32(kEmberHi), 12);
    }
}

std::string find_example(const char* name) {
    namespace fs = std::filesystem;
    fs::path exe = upath(exe_dir());
    for (const fs::path& base : {exe / "examples", exe / ".." / "examples", exe / ".." / ".." / "examples", fs::path("examples")}) {
        std::error_code ec;
        fs::path p = base / name;
        if (fs::exists(p, ec)) return fs::weakly_canonical(p, ec).u8string();
    }
    return "";
}

// ---------------------------------------------------------------- actions

void finalize_spec() {
    if (A.data.kind == DATA_TEXT) {
        A.spec.kind = MODEL_GPT;
        A.spec.vocab = (int)A.data.vocab.size();
        A.spec.ctx = std::max(8, std::min(A.spec.ctx, (int)A.data.n_train - 2));
    } else if (A.data.kind == DATA_TABLE) {
        A.spec.kind = MODEL_MLP;
        A.spec.in_dim = A.data.n_feat;
        A.spec.regression = A.data.regression;
        A.spec.out_dim = A.data.regression ? 1 : (int)A.data.labels.size();
    }
    A.spec.lora_rank = 0;
}

void auto_configure() {
    if (A.data.kind == DATA_NONE) return;
    A.sys = detect_system();  // free memory changes over time
    A.plan = autotune(A.sys, A.data);
    A.spec = A.plan.spec;
    A.cfg = A.plan.cfg;
    A.has_plan = true;
}

void reset_generation() {
    A.gen_left = 0;
    A.gen_ids.clear();
    A.gen_out.clear();
    A.pred.clear();
}

void build_model() {
    if (A.data.kind == DATA_NONE) return;
    finalize_spec();
    A.model = std::make_shared<Model>(A.spec, (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count());
    if (A.data.kind == DATA_TEXT) {
        A.model->vocab = A.data.vocab;
    } else {
        A.model->vocab = A.data.labels;
        A.model->norm_mean = A.data.mean;
        A.model->norm_std = A.data.stdv;
        for (int c = 0; c < A.data.n_cols; ++c)
            if (c != A.data.target_col) A.model->feat_names.push_back(A.data.header[c]);
    }
    reset_generation();
}

// True if the current model can keep training on the loaded dataset.
bool model_matches_data() {
    if (!A.model || A.data.kind == DATA_NONE) return false;
    const ModelSpec& s = A.model->spec;
    if (s.kind == MODEL_GPT) return A.data.kind == DATA_TEXT;
    return A.data.kind == DATA_TABLE && s.in_dim == A.data.n_feat && s.regression == A.data.regression &&
           (s.regression || s.out_dim == (int)A.data.labels.size());
}

bool load_data_impl(const std::string& path, int kind, int target_col) {
    if (busy()) return false;
    std::string err;
    Dataset d;
    bool ok = kind == 1 ? d.load_table(path, target_col, &err) : d.load_text(path, &err);
    A.data_ok = ok;
    if (!ok) { A.data_msg = err; return false; }
    A.data = std::move(d);
    A.data_kind = kind;
    A.data_msg.clear();
    std::snprintf(A.data_path, sizeof A.data_path, "%s", path.c_str());
    if (model_matches_data()) {
        // Keep the model (fine-tuning): text is re-encoded with its vocabulary.
        if (A.data.kind == DATA_TEXT) A.data.encode_with(A.model->vocab);
        TrainConfig keep = A.cfg;
        A.plan = autotune(A.sys, A.data);
        A.cfg = keep;
    } else {
        A.model.reset();
        reset_generation();
        auto_configure();
    }
    return true;
}

void reset_model() {
    if (busy()) return;
    A.model.reset();
    reset_generation();
    if (A.data.kind == DATA_TEXT) {
        std::string err;
        A.data.load_text(A.data.path, &err);  // restore the dataset's own vocabulary
    }
    auto_configure();
}

void begin_training() {
    if (busy() || A.data.kind == DATA_NONE) return;
    if (!A.model || !model_matches_data()) build_model();
    A.gen_left = 0;
    A.cfg.seed = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
    A.trainer.start(A.model, &A.data, A.cfg);
}

std::string settings_file() { return (upath(config_dir()) / "settings.cfg").u8string(); }

// ------------------------------------------------------------------ pages

void need_data_notice() {
    Muted(tr(S::NoData));
    if (ImGui::Button(tr(S::GoData))) A.page = PAGE_DATA;
}

void page_home() {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float sz = 84 * A.sc;
    draw_logo(dl, p, sz);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + sz + 18 * A.sc);
    ImGui::BeginGroup();
    ImGui::Dummy(ImVec2(0, 8 * A.sc));
    Title("Kor Studio");
    ImGui::TextColored(kEmberHi, "%s", tr(S::Tagline));
    ImGui::EndGroup();
    ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + sz + 16 * A.sc));
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 820 * A.sc);
    ImGui::TextWrapped("%s", tr(S::HomeIntro));
    ImGui::PopTextWrapPos();
    ImGui::Dummy(ImVec2(0, 8 * A.sc));

    float w = ImGui::GetContentRegionAvail().x;
    float colw = (w - 12 * A.sc) * 0.5f;
    ImGui::BeginChild("##left", ImVec2(colw, 0), ImGuiChildFlags_AutoResizeY);
    if (BeginCard("sys")) {
        ImGui::SeparatorText(tr(S::YourSystem));
        char b[256];
        Row(tr(S::Cpu), A.sys.cpu.c_str());
        std::snprintf(b, sizeof b, tr(S::ThreadsFmt), A.sys.logical, recommended_threads(A.sys));
        Row(tr(S::Threads), b);
        std::snprintf(b, sizeof b, tr(S::RamFmt), A.sys.ram_total_gb, A.sys.ram_free_gb);
        Row(tr(S::Ram), b);
        Row(tr(S::Simd), K().name, &kGood);
        for (auto& g : A.sys.gpus) Row(tr(S::Gpu), g.c_str());
        ImGui::Dummy(ImVec2(0, 2 * A.sc));
        Muted(tr(S::GpuNote));
    }
    EndCard();
    if (BeginCard("quick")) {
        ImGui::SeparatorText(tr(S::QuickStart));
        for (S s : {S::Step1, S::Step2, S::Step3, S::Step4}) ImGui::TextWrapped("%s", tr(s));
        ImGui::Dummy(ImVec2(0, 4 * A.sc));
        if (PrimaryButton(tr(S::Start), ImVec2(160 * A.sc, 0))) A.page = PAGE_DATA;
    }
    EndCard();
    ImGui::EndChild();
    ImGui::SameLine(0, 12 * A.sc);
    ImGui::BeginChild("##right", ImVec2(colw, 0), ImGuiChildFlags_AutoResizeY);
    if (BeginCard("light")) {
        ImGui::SeparatorText(tr(S::LightTitle));
        for (S s : {S::Light1, S::Light2, S::Light3, S::Light4, S::Light5}) {
            ImGui::TextColored(kEmber, "%s", "\xE2\x80\xA2");  // bullet
            ImGui::SameLine();
            ImGui::TextWrapped("%s", tr(s));
        }
    }
    EndCard();
    ImGui::EndChild();
}

void page_data() {
    Title(tr(S::NavData));
    Muted(tr(S::DataHelp));
    ImGui::Dummy(ImVec2(0, 4 * A.sc));
    ImGui::BeginDisabled(busy());
    ImGui::RadioButton(tr(S::KindText), &A.data_kind, 0);
    ImGui::SameLine();
    ImGui::RadioButton(tr(S::KindTable), &A.data_kind, 1);
    ImGui::SetNextItemWidth(-260 * A.sc);
    bool enter = ImGui::InputTextWithHint("##path", tr(S::PathHint), A.data_path, sizeof A.data_path, ImGuiInputTextFlags_EnterReturnsTrue);
    if (has_native_dialogs()) {
        ImGui::SameLine();
        if (ImGui::Button(tr(S::Browse))) {
            std::string p = open_file_dialog(DLG_DATA);
            if (!p.empty()) {
                std::snprintf(A.data_path, sizeof A.data_path, "%s", p.c_str());
                std::string ext = p.size() > 4 ? p.substr(p.size() - 4) : "";
                if (ext == ".csv" || ext == ".tsv") A.data_kind = 1;
                else if (ext == ".txt") A.data_kind = 0;
                enter = true;
            }
        }
    }
    ImGui::SameLine();
    if (PrimaryButton(tr(S::Load)) || enter) load_data_impl(A.data_path, A.data_kind, -1);
    for (int k = 0; k < 2; ++k) {
        if (k) ImGui::SameLine();
        if (ImGui::Button(tr(k ? S::ExampleTable : S::ExampleText))) {
            std::string p = find_example(k ? "spiral.csv" : "tiny_text.txt");
            if (p.empty()) { A.data_ok = false; A.data_msg = tr(S::ExampleMissing); }
            else load_data_impl(p, k, -1);
        }
    }
    ImGui::EndDisabled();
    if (busy()) Muted(tr(S::Busy));
    if (!A.data_ok && !A.data_msg.empty()) {
        char b[512];
        std::snprintf(b, sizeof b, tr(S::Failed), A.data_msg.c_str());
        ImGui::TextColored(kBad, "%s", b);
    }
    ImGui::Dummy(ImVec2(0, 6 * A.sc));
    if (A.data.kind == DATA_NONE) return;

    char b[256];
    if (BeginCard("dinfo")) {
        ImGui::TextColored(kGood, "%s", tr(S::Loaded));
        ImGui::SameLine();
        ImGui::TextColored(kMuted, "%s", A.data.path.c_str());
        if (A.data.kind == DATA_TEXT) {
            Row(tr(S::Characters), fmt_count((double)A.data.ids.size()).c_str());
            std::snprintf(b, sizeof b, "%d", (int)A.data.vocab.size());
            Row(tr(S::VocabSize), b);
            std::snprintf(b, sizeof b, "%s / %s", fmt_count((double)A.data.train_items()).c_str(), fmt_count((double)A.data.val_items()).c_str());
            Row(tr(S::TrainVal), b);
        } else {
            std::snprintf(b, sizeof b, "%d", (int)(A.data.train_idx.size() + A.data.val_idx.size()));
            Row(tr(S::Rows), b);
            std::snprintf(b, sizeof b, "%d", A.data.n_feat);
            Row(tr(S::Features), b);
            if (A.data.regression) Row(tr(S::Task), tr(S::Regression));
            else {
                std::snprintf(b, sizeof b, tr(S::Classification), (int)A.data.labels.size());
                Row(tr(S::Task), b);
            }
            std::snprintf(b, sizeof b, "%d / %d", (int)A.data.train_idx.size(), (int)A.data.val_idx.size());
            Row(tr(S::TrainVal), b);
            ImGui::TextColored(kMuted, "%s", tr(S::Target));
            ImGui::SameLine(230 * A.sc);
            ImGui::SetNextItemWidth(240 * A.sc);
            ImGui::BeginDisabled(busy());
            if (ImGui::BeginCombo("##target", A.data.header[A.data.target_col].c_str())) {
                for (int c = 0; c < A.data.n_cols; ++c)
                    if (ImGui::Selectable(A.data.header[c].c_str(), c == A.data.target_col) && c != A.data.target_col) {
                        std::string path = A.data.path;
                        load_data_impl(path, 1, c);
                        break;
                    }
                ImGui::EndCombo();
            }
            ImGui::EndDisabled();
            if (A.data.bad_cells > 0) {
                std::snprintf(b, sizeof b, tr(S::BadCells), A.data.bad_cells);
                ImGui::TextColored(kEmberHi, "%s", b);
            }
        }
    }
    EndCard();
    if (A.data.kind == DATA_NONE) return;

    ImGui::SeparatorText(tr(S::Preview));
    if (A.data.kind == DATA_TEXT) {
        size_t n = std::min<size_t>(A.data.raw.size(), 1200);
        while (n < A.data.raw.size() && ((unsigned char)A.data.raw[n] & 0xC0) == 0x80) ++n;  // do not cut a UTF-8 sequence
        ImGui::BeginChild("##ptext", ImVec2(0, 0), ImGuiChildFlags_Borders);
        ImGui::PushTextWrapPos(0);
        ImGui::TextUnformatted(A.data.raw.data(), A.data.raw.data() + n);
        ImGui::PopTextWrapPos();
        ImGui::EndChild();
    } else if (ImGui::BeginTable("##ptable", std::min(A.data.n_cols, 60),
                                 ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollX | ImGuiTableFlags_SizingFixedFit)) {
        int nc = std::min(A.data.n_cols, 60);
        for (int c = 0; c < nc; ++c) ImGui::TableSetupColumn(A.data.header[c].c_str());
        ImGui::TableHeadersRow();
        for (auto& r : A.data.preview) {
            ImGui::TableNextRow();
            for (int c = 0; c < nc; ++c) {
                ImGui::TableSetColumnIndex(c);
                if (c == A.data.target_col) ImGui::TextColored(kEmberHi, "%s", r[c].c_str());
                else ImGui::TextUnformatted(r[c].c_str());
            }
        }
        ImGui::EndTable();
    }
}

void draw_diagram(const ModelSpec& s) {
    struct Box { std::string top, sub; ImU32 col; };
    std::vector<Box> boxes;
    const ImU32 cIO = IM_COL32(70, 60, 54, 255), cDense = IM_COL32(190, 86, 16, 255), cAct = IM_COL32(52, 96, 140, 255),
                cNorm = IM_COL32(70, 110, 78, 255);
    char b[64];
    if (s.kind == MODEL_GPT) {
        std::snprintf(b, sizeof b, "%d", s.ctx);
        boxes.push_back({tr(S::Input), b, cIO});
        std::snprintf(b, sizeof b, "%d x %d", s.vocab, s.dim);
        boxes.push_back({"Embedding", b, cNorm});
        for (int i = 0; i < s.nlayer && i < 12; ++i) {
            std::snprintf(b, sizeof b, "%d heads", s.heads);
            boxes.push_back({"Attention", b, cDense});
            std::snprintf(b, sizeof b, "%d", s.dim * 4);
            boxes.push_back({"MLP", b, cAct});
        }
        boxes.push_back({"LayerNorm", "", cNorm});
        std::snprintf(b, sizeof b, "%d", s.vocab);
        boxes.push_back({tr(S::Output), b, cIO});
    } else {
        std::snprintf(b, sizeof b, "%d", s.in_dim);
        boxes.push_back({tr(S::Input), b, cIO});
        for (auto& l : s.layers) {
            if (l.type == L_DENSE) { std::snprintf(b, sizeof b, "%d", l.units); boxes.push_back({layer_name(l.type), b, cDense}); }
            else if (l.type == L_DROPOUT) { std::snprintf(b, sizeof b, "%.0f%%", l.p * 100); boxes.push_back({layer_name(l.type), b, cNorm}); }
            else boxes.push_back({layer_name(l.type), "", l.type == L_LAYERNORM ? cNorm : cAct});
        }
        std::snprintf(b, sizeof b, "%d", s.out_dim);
        boxes.push_back({tr(S::Output), b, cIO});
    }
    float h = 58 * A.sc, gap = 26 * A.sc, pad = 12 * A.sc;
    const ImGuiStyle& style = ImGui::GetStyle();
    ImGui::BeginChild("##diagram", ImVec2(0, h + style.WindowPadding.y * 2 + style.ScrollbarSize + 4 * A.sc),
                      ImGuiChildFlags_Borders, ImGuiWindowFlags_HorizontalScrollbar);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float x = p.x;
    for (size_t i = 0; i < boxes.size(); ++i) {
        ImVec2 t1 = ImGui::CalcTextSize(boxes[i].top.c_str()), t2 = ImGui::CalcTextSize(boxes[i].sub.c_str());
        float w = std::max(t1.x, t2.x) + pad * 2;
        dl->AddRectFilled(ImVec2(x, p.y), ImVec2(x + w, p.y + h), boxes[i].col, 8 * A.sc);
        float ty = boxes[i].sub.empty() ? p.y + (h - t1.y) * 0.5f : p.y + h * 0.5f - t1.y - 1;
        dl->AddText(ImVec2(x + (w - t1.x) * 0.5f, ty), IM_COL32(255, 255, 255, 255), boxes[i].top.c_str());
        if (!boxes[i].sub.empty())
            dl->AddText(ImVec2(x + (w - t2.x) * 0.5f, p.y + h * 0.5f + 1), IM_COL32(255, 226, 190, 255), boxes[i].sub.c_str());
        x += w;
        if (i + 1 < boxes.size()) {
            float y = p.y + h * 0.5f;
            ImU32 c = IM_COL32(150, 140, 132, 255);
            dl->AddLine(ImVec2(x + 4 * A.sc, y), ImVec2(x + gap - 5 * A.sc, y), c, 1.5f * A.sc);
            dl->AddTriangleFilled(ImVec2(x + gap - 3 * A.sc, y), ImVec2(x + gap - 9 * A.sc, y - 4 * A.sc), ImVec2(x + gap - 9 * A.sc, y + 4 * A.sc), c);
            x += gap;
        }
    }
    ImGui::Dummy(ImVec2(x - p.x, h));
    ImGui::EndChild();
}

void page_model() {
    Title(tr(S::NavModel));
    Muted(tr(S::ModelHelp));
    ImGui::Dummy(ImVec2(0, 4 * A.sc));
    char b[256];

    if (A.model) {  // ---- a model exists: show it, offer LoRA and reset
        if (BeginCard("built")) {
            ImGui::TextColored(kGood, "%s", tr(S::Built));
            ImGui::SameLine();
            ImGui::TextColored(kMuted, "%s", tr(A.model->spec.kind == MODEL_GPT ? S::TextModel : S::TableModel));
            size_t total, train;
            {
                std::lock_guard<std::mutex> lk(A.trainer.model_mu);
                total = A.model->param_count(false);
                train = A.model->param_count(true);
            }
            std::snprintf(b, sizeof b, "%s  (%s %s)", fmt_count((double)total).c_str(), fmt_count((double)train).c_str(), tr(S::Trainable));
            Row(tr(S::Params), b);
            draw_diagram(A.model->spec);
            Muted(tr(S::ResetHint));
            ImGui::BeginDisabled(busy());
            if (ImGui::Button(tr(S::ResetModel))) reset_model();
            ImGui::EndDisabled();
        }
        EndCard();
        if (A.model && BeginCard("lora")) {
            ImGui::SeparatorText(tr(S::LoraTitle));
            Muted(tr(S::LoraHelp));
            ImGui::BeginDisabled(busy());
            if (A.model->spec.lora_rank == 0) {
                ImGui::SetNextItemWidth(140 * A.sc);
                ImGui::SliderInt(tr(S::Rank), &A.lora_rank, 1, 32);
                ImGui::SameLine();
                if (ImGui::Button(tr(S::EnableLora))) A.model->enable_lora(A.lora_rank);
            } else {
                std::snprintf(b, sizeof b, "%s: %d", tr(S::Rank), A.model->spec.lora_rank);
                ImGui::TextColored(kEmberHi, "%s", b);
                ImGui::SameLine();
                if (ImGui::Button(tr(S::MergeLora))) A.model->merge_lora();
            }
            ImGui::EndDisabled();
        }
        if (A.model) EndCard();
        return;
    }

    if (A.data.kind == DATA_NONE) { need_data_notice(); return; }
    finalize_spec();

    if (PrimaryButton(tr(S::AutoBtn))) auto_configure();
    if (A.has_plan) {
        ImGui::SameLine();
        std::snprintf(b, sizeof b, tr(S::AutoDone), tier_name(A.plan.tier));
        ImGui::TextColored(kMuted, "%s", b);
    }
    ImGui::Dummy(ImVec2(0, 4 * A.sc));

    if (BeginCard("arch")) {
        ImGui::SeparatorText(tr(S::Arch));
        ModelSpec& s = A.spec;
        if (s.kind == MODEL_GPT) {
            int per = std::max(4, s.dim / std::max(1, s.heads));
            ImGui::PushItemWidth(320 * A.sc);
            ImGui::SliderInt(tr(S::Heads), &s.heads, 1, 12);
            ImGui::SliderInt(tr(S::HeadDim), &per, 8, 64);
            ImGui::SliderInt(tr(S::Layers), &s.nlayer, 1, 12);
            int max_ctx = std::max(8, std::min(512, (int)A.data.n_train - 2));
            ImGui::SliderInt(tr(S::Context), &s.ctx, 8, max_ctx);
            ImGui::PopItemWidth();
            s.heads = std::max(1, std::min(12, s.heads));
            s.nlayer = std::max(1, std::min(12, s.nlayer));
            s.dim = s.heads * std::max(4, std::min(64, per));
            s.ctx = std::max(8, std::min(max_ctx, s.ctx));
            std::snprintf(b, sizeof b, tr(S::Width), s.dim);
            ImGui::TextColored(kMuted, "%s", b);
        } else {
            int remove = -1, up = -1;
            for (int i = 0; i < (int)s.layers.size(); ++i) {
                LayerSpec& l = s.layers[i];
                ImGui::PushID(i);
                ImGui::TextColored(kMuted, "%2d", i + 1);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(190 * A.sc);
                if (ImGui::BeginCombo("##type", layer_name(l.type))) {
                    for (int t = 0; t < L_TYPE_COUNT; ++t)
                        if (ImGui::Selectable(layer_name(t), t == l.type)) l.type = t;
                    ImGui::EndCombo();
                }
                ImGui::SameLine();
                if (l.type == L_DENSE) {
                    ImGui::SetNextItemWidth(150 * A.sc);
                    ImGui::InputInt(tr(S::Units), &l.units, 8, 64);
                    l.units = std::max(1, std::min(4096, l.units));
                } else if (l.type == L_DROPOUT) {
                    ImGui::SetNextItemWidth(150 * A.sc);
                    ImGui::SliderFloat("p", &l.p, 0.f, 0.8f, "%.2f");
                } else {
                    ImGui::Dummy(ImVec2(150 * A.sc, 1));
                }
                ImGui::SameLine(520 * A.sc);
                if (ImGui::SmallButton("^") && i > 0) up = i;
                ImGui::SameLine();
                if (ImGui::SmallButton("x")) remove = i;
                ImGui::PopID();
            }
            if (up > 0) std::swap(s.layers[up], s.layers[up - 1]);
            if (remove >= 0) s.layers.erase(s.layers.begin() + remove);
            ImGui::SetNextItemWidth(190 * A.sc);
            if (ImGui::BeginCombo("##add", layer_name(A.add_type))) {
                for (int t = 0; t < L_TYPE_COUNT; ++t)
                    if (ImGui::Selectable(layer_name(t), t == A.add_type)) A.add_type = t;
                ImGui::EndCombo();
            }
            ImGui::SameLine();
            if (ImGui::Button(tr(S::AddLayer)) && s.layers.size() < 64) {
                LayerSpec l;
                l.type = A.add_type;
                s.layers.push_back(l);
            }
        }
        ImGui::Dummy(ImVec2(0, 4 * A.sc));
        draw_diagram(s);
    }
    EndCard();

    if (BeginCard("fit")) {
        double mb = (double)estimate_train_bytes(A.spec, A.cfg.batch) / (1024.0 * 1024.0);
        double budget = A.has_plan ? A.plan.budget_mb : 512;
        Row(tr(S::Params), fmt_count((double)estimate_params(A.spec)).c_str());
        char bud[96];
        std::snprintf(bud, sizeof bud, tr(S::Budget), budget);
        std::snprintf(b, sizeof b, "%s  (%s)", fmt_mb(mb).c_str(), bud);
        Row(tr(S::EstRam), b);
        if (mb <= budget) ImGui::TextColored(kGood, "%s", tr(S::Fits));
        else ImGui::TextColored(kBad, "%s", tr(S::TooBig));
    }
    EndCard();
    if (PrimaryButton(tr(S::BuildModel))) build_model();
}

void plot_loss(const TrainStats& st, float height) {
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImVec2 size(ImGui::GetContentRegionAvail().x, height);
    ImGui::InvisibleButton("##plot", size);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p1(p0.x + size.x, p0.y + size.y);
    dl->AddRectFilled(p0, p1, IM_COL32(20, 17, 15, 255), 8 * A.sc);
    dl->AddRect(p0, p1, IM_COL32(64, 56, 50, 255), 8 * A.sc);
    if (st.train_hist.size() < 2) {
        ImVec2 ts = ImGui::CalcTextSize(tr(S::PlotEmpty));
        dl->AddText(ImVec2(p0.x + (size.x - ts.x) * 0.5f, p0.y + (size.y - ts.y) * 0.5f), ImGui::GetColorU32(kMuted), tr(S::PlotEmpty));
        return;
    }
    float lo = 1e30f, hi = -1e30f;
    for (float v : st.train_hist) { lo = std::min(lo, v); hi = std::max(hi, v); }
    for (float v : st.val_hist) { lo = std::min(lo, v); hi = std::max(hi, v); }
    if (hi - lo < 1e-6f) hi = lo + 1e-6f;
    float padl = 58 * A.sc, padr = 14 * A.sc, padt = 30 * A.sc, padb = 14 * A.sc;
    ImVec2 a(p0.x + padl, p0.y + padt), bq(p1.x - padr, p1.y - padb);
    int total = std::max(st.total, (int)st.train_hist.size());
    auto X = [&](float step) { return a.x + (bq.x - a.x) * (step - 1) / std::max(1, total - 1); };
    auto Y = [&](float v) { return bq.y - (bq.y - a.y) * (v - lo) / (hi - lo); };
    char b[32];
    for (int g = 0; g <= 4; ++g) {
        float v = lo + (hi - lo) * g / 4.f, y = Y(v);
        dl->AddLine(ImVec2(a.x, y), ImVec2(bq.x, y), IM_COL32(48, 42, 38, 255));
        std::snprintf(b, sizeof b, "%.3f", v);
        ImVec2 ts = ImGui::CalcTextSize(b);
        dl->AddText(ImVec2(a.x - ts.x - 6 * A.sc, y - ts.y * 0.5f), ImGui::GetColorU32(kMuted), b);
    }
    int n = (int)st.train_hist.size();
    int stride = std::max(1, n / std::max(1, (int)(bq.x - a.x)));
    for (int i = 0; i < n; i += stride) dl->PathLineTo(ImVec2(X((float)(i + 1)), Y(st.train_hist[i])));
    dl->PathLineTo(ImVec2(X((float)n), Y(st.train_hist[n - 1])));
    dl->PathStroke(ImGui::GetColorU32(kEmber), 0, 2.f * A.sc);
    for (size_t i = 0; i < st.val_hist.size(); ++i) dl->PathLineTo(ImVec2(X((float)st.val_steps[i]), Y(st.val_hist[i])));
    dl->PathStroke(ImGui::GetColorU32(kBlue), 0, 2.f * A.sc);
    for (size_t i = 0; i < st.val_hist.size(); ++i)
        dl->AddCircleFilled(ImVec2(X((float)st.val_steps[i]), Y(st.val_hist[i])), 3 * A.sc, ImGui::GetColorU32(kBlue));
    float lx = a.x, ly = p0.y + 8 * A.sc;
    for (int k = 0; k < 2; ++k) {
        const char* name = tr(k ? S::ValLoss : S::Loss);
        dl->AddRectFilled(ImVec2(lx, ly + 5 * A.sc), ImVec2(lx + 14 * A.sc, ly + 9 * A.sc), ImGui::GetColorU32(k ? kBlue : kEmber));
        dl->AddText(ImVec2(lx + 20 * A.sc, ly - 2 * A.sc), ImGui::GetColorU32(ImGuiCol_Text), name);
        lx += ImGui::CalcTextSize(name).x + 44 * A.sc;
    }
}

void page_train() {
    Title(tr(S::NavTrain));
    if (A.data.kind == DATA_NONE) { need_data_notice(); return; }
    TrainStats st = A.trainer.stats();
    bool run = busy();
    char b[256];

    if (BeginCard("cfg")) {
        ImGui::SeparatorText(tr(S::Settings));
        ImGui::BeginDisabled(run);
        float w = (ImGui::GetContentRegionAvail().x - 24 * A.sc) / 2;
        ImGui::BeginGroup();
        ImGui::PushItemWidth(170 * A.sc);
        ImGui::InputInt(tr(S::Steps), &A.cfg.steps, 100, 1000);
        ImGui::InputInt(tr(S::BatchSize), &A.cfg.batch, 1, 8);
        ImGui::InputInt(tr(S::Accum), &A.cfg.accum, 1, 2);
        ImGui::InputInt(tr(S::EvalEvery), &A.cfg.eval_every, 10, 100);
        ImGui::PopItemWidth();
        ImGui::EndGroup();
        ImGui::SameLine(w + 24 * A.sc);
        ImGui::BeginGroup();
        ImGui::PushItemWidth(170 * A.sc);
        ImGui::InputFloat(tr(S::LR), &A.cfg.lr, 0, 0, "%.5f");
        ImGui::InputFloat(tr(S::WD), &A.cfg.weight_decay, 0, 0, "%.4f");
        ImGui::SliderInt(tr(S::ThreadsUse), &A.cfg.threads, 1, std::max(1, A.sys.logical));
        const char* opts[] = {"AdamW", "SGD + momentum"};
        ImGui::Combo(tr(S::Optimizer), &A.cfg.optimizer, opts, 2);
        ImGui::PopItemWidth();
        ImGui::EndGroup();
        A.cfg.steps = std::max(1, std::min(10000000, A.cfg.steps));
        A.cfg.batch = std::max(1, std::min(4096, A.cfg.batch));
        A.cfg.accum = std::max(1, std::min(256, A.cfg.accum));
        A.cfg.eval_every = std::max(1, A.cfg.eval_every);
        A.cfg.lr = std::max(1e-7f, std::min(1.f, A.cfg.lr));
        A.cfg.weight_decay = std::max(0.f, std::min(1.f, A.cfg.weight_decay));
        A.cfg.threads = std::max(1, std::min(std::max(1, A.sys.logical), A.cfg.threads));
        A.cfg.warmup = std::min(100, A.cfg.steps / 10);
        ImGui::EndDisabled();
        const ModelSpec& ms = A.model && model_matches_data() ? A.model->spec : A.spec;
        double mb = (double)estimate_train_bytes(ms, A.cfg.batch) / (1024.0 * 1024.0);
        char eff[96];
        std::snprintf(eff, sizeof eff, tr(S::EffBatch), A.cfg.batch * A.cfg.accum);
        std::snprintf(b, sizeof b, "%s   |   %s: %s", eff, tr(S::EstRam), fmt_mb(mb).c_str());
        ImGui::TextColored(kMuted, "%s", b);
    }
    EndCard();

    ImVec2 bs(190 * A.sc, 34 * A.sc);
    if (!run) {
        if (PrimaryButton(tr(A.model && model_matches_data() ? S::ContinueTrain : S::StartTrain), bs)) begin_training();
    } else {
        if (PrimaryButton(tr(A.trainer.paused() ? S::Resume : S::Pause), bs)) A.trainer.pause(!A.trainer.paused());
        ImGui::SameLine();
        if (ImGui::Button(tr(S::Stop), ImVec2(120 * A.sc, bs.y))) A.trainer.stop();
    }
    ImGui::SameLine(0, 18 * A.sc);
    static const S state_names[] = {S::StIdle, S::StRunning, S::StPaused, S::StDone, S::StStopped, S::StError};
    const ImVec4& sc = st.state == TS_ERROR ? kBad : st.state == TS_DONE ? kGood : st.state == TS_RUNNING ? kEmberHi : kMuted;
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(sc, "%s", tr(state_names[st.state]));
    if (st.state == TS_ERROR) {
        ImGui::SameLine();
        ImGui::TextColored(kBad, "%s", st.error.c_str());
    }
    ImGui::Dummy(ImVec2(0, 4 * A.sc));

    std::snprintf(b, sizeof b, "%d / %d", st.step, st.total);
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, kEmber);
    ImGui::ProgressBar(st.total > 0 ? (float)st.step / st.total : 0.f, ImVec2(-1, 22 * A.sc), b);
    ImGui::PopStyleColor();

    if (ImGui::BeginTable("##metrics", 6, ImGuiTableFlags_SizingStretchSame)) {
        auto cell = [&](const char* label, const char* value, const ImVec4& col) {
            ImGui::TableNextColumn();
            ImGui::TextColored(kMuted, "%s", label);
            ImGui::TextColored(col, "%s", value);
        };
        const ImVec4 white = ImGui::GetStyleColorVec4(ImGuiCol_Text);
        std::snprintf(b, sizeof b, st.step ? "%.4f" : "-", st.loss);
        cell(tr(S::Loss), b, kEmber);
        std::snprintf(b, sizeof b, st.val_loss >= 0 ? "%.4f" : "-", st.val_loss);
        cell(tr(S::ValLoss), b, kBlue);
        std::snprintf(b, sizeof b, st.val_acc >= 0 ? "%.1f%%" : "-", st.val_acc * 100);
        cell(tr(S::Accuracy), b, white);
        if (st.step) std::snprintf(b, sizeof b, tr(A.data.kind == DATA_TEXT ? S::TokPerSec : S::SampPerSec), st.items_per_sec);
        else std::snprintf(b, sizeof b, "-");
        cell(tr(S::Speed), b, white);
        cell(tr(S::Elapsed), fmt_time(st.elapsed).c_str(), white);
        cell(tr(S::Eta), st.state == TS_RUNNING || st.state == TS_PAUSED ? fmt_time(st.eta).c_str() : "-", white);
        ImGui::EndTable();
    }
    ImGui::Dummy(ImVec2(0, 4 * A.sc));
    plot_loss(st, std::max(180 * A.sc, ImGui::GetContentRegionAvail().y - 8 * A.sc));
}

// Generates a few characters per frame so the interface never blocks; uses
// try_lock so it interleaves with training steps.
void pump_generation() {
    if (A.gen_left <= 0 || !A.model) return;
    std::unique_lock<std::mutex> lk(A.trainer.model_mu, std::try_to_lock);
    if (!lk.owns_lock()) return;
    auto t0 = std::chrono::steady_clock::now();
    while (A.gen_left > 0) {
        int t = A.model->next_token(A.gen_ids, A.temperature, A.gen_rng);
        A.gen_ids.push_back(t);
        if (t >= 0 && t < (int)A.model->vocab.size()) A.gen_out += A.model->vocab[t];
        --A.gen_left;
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > 0.012) break;
    }
}

void page_test() {
    Title(tr(S::NavTest));
    if (!A.model) {
        Muted(tr(S::NoModel));
        return;
    }
    if (A.model->spec.kind == MODEL_GPT) {
        ImGui::TextColored(kMuted, "%s", tr(S::Prompt));
        ImGui::InputTextMultiline("##prompt", A.prompt, sizeof A.prompt, ImVec2(-1, 70 * A.sc));
        ImGui::PushItemWidth(260 * A.sc);
        ImGui::SliderFloat(tr(S::Temperature), &A.temperature, 0.f, 1.5f, "%.2f");
        ImGui::SliderInt(tr(S::Length), &A.gen_len, 20, 2000);
        ImGui::PopItemWidth();
        if (A.gen_left > 0) {
            if (ImGui::Button(tr(S::Stop), ImVec2(160 * A.sc, 0))) A.gen_left = 0;
        } else if (PrimaryButton(tr(S::Generate), ImVec2(160 * A.sc, 0))) {
            A.gen_ids = A.model->encode(A.prompt);
            if (A.gen_ids.empty()) A.gen_ids.push_back(A.gen_rng.below(std::max(1, A.model->spec.vocab)));
            A.gen_out = A.prompt;
            A.gen_left = A.gen_len;
        }
        pump_generation();
        ImGui::SeparatorText(tr(S::Result));
        ImGui::BeginChild("##out", ImVec2(0, 0), ImGuiChildFlags_Borders);
        ImGui::PushTextWrapPos(0);
        ImGui::TextUnformatted(A.gen_out.data(), A.gen_out.data() + A.gen_out.size());
        ImGui::PopTextWrapPos();
        if (A.gen_left > 0) ImGui::SetScrollHereY(1.f);
        ImGui::EndChild();
    } else {
        const ModelSpec& s = A.model->spec;
        if ((int)A.feat_in.size() != s.in_dim) {
            A.feat_in.assign(s.in_dim, 0.f);
            for (int j = 0; j < s.in_dim && j < (int)A.model->norm_mean.size(); ++j) A.feat_in[j] = A.model->norm_mean[j];
            A.pred.clear();
        }
        ImGui::SeparatorText(tr(S::InputsLbl));
        ImGui::BeginChild("##feat", ImVec2(0, std::min(300.f, 34.f * s.in_dim + 16) * A.sc), ImGuiChildFlags_Borders);
        ImGui::PushItemWidth(200 * A.sc);
        for (int j = 0; j < s.in_dim; ++j) {
            ImGui::PushID(j);
            std::string name = j < (int)A.model->feat_names.size() ? A.model->feat_names[j] : "x" + std::to_string(j + 1);
            ImGui::InputFloat(name.c_str(), &A.feat_in[j], 0, 0, "%.4f");
            ImGui::PopID();
        }
        ImGui::PopItemWidth();
        ImGui::EndChild();
        if (PrimaryButton(tr(S::Predict), ImVec2(160 * A.sc, 0))) {
            std::lock_guard<std::mutex> lk(A.trainer.model_mu);
            A.pred = A.model->predict(A.feat_in);
        }
        if (!A.pred.empty()) {
            ImGui::SeparatorText(tr(S::Result));
            if (s.regression) {
                if (A.title_font) ImGui::PushFont(A.title_font);
                ImGui::TextColored(kEmberHi, "%.5g", A.pred[0]);
                if (A.title_font) ImGui::PopFont();
            } else {
                size_t best = 0;
                for (size_t i = 1; i < A.pred.size(); ++i)
                    if (A.pred[i] > A.pred[best]) best = i;
                for (size_t i = 0; i < A.pred.size(); ++i) {
                    std::string name = i < A.model->vocab.size() ? A.model->vocab[i] : std::to_string(i);
                    char pb[32];
                    std::snprintf(pb, sizeof pb, "%.1f%%", A.pred[i] * 100);
                    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, i == best ? kEmber : ImVec4(0.35f, 0.31f, 0.28f, 1));
                    ImGui::ProgressBar(A.pred[i], ImVec2(360 * A.sc, 0), pb);
                    ImGui::PopStyleColor();
                    ImGui::SameLine();
                    ImGui::TextUnformatted(name.c_str());
                }
            }
        }
    }
}

void page_files() {
    Title(tr(S::NavFiles));
    Muted(tr(S::FilesHelp));
    ImGui::Dummy(ImVec2(0, 4 * A.sc));
    ImGui::SetNextItemWidth(-140 * A.sc);
    ImGui::InputTextWithHint("##mpath", tr(S::PathHint), A.model_path, sizeof A.model_path);
    ImGui::Dummy(ImVec2(0, 4 * A.sc));

    if (BeginCard("save")) {
        ImGui::SeparatorText(tr(S::SaveModel));
        if (!A.model) Muted(tr(S::NoModel));
        ImGui::BeginDisabled(!A.model);
        ImGui::Checkbox(tr(S::Half), &A.half);
        bool go = false;
        if (has_native_dialogs()) {
            if (ImGui::Button(tr(S::Browse))) {
                std::string p = save_file_dialog();
                if (!p.empty()) { std::snprintf(A.model_path, sizeof A.model_path, "%s", p.c_str()); go = true; }
            }
            ImGui::SameLine();
        }
        if (PrimaryButton(tr(S::Save), ImVec2(140 * A.sc, 0))) go = true;
        if (go && A.model && A.model_path[0]) {
            std::string err;
            std::lock_guard<std::mutex> lk(A.trainer.model_mu);
            A.file_ok = A.model->save(A.model_path, A.half, &err);
            A.file_msg = A.file_ok ? tr(S::Saved) : err;
        }
        ImGui::EndDisabled();
    }
    EndCard();

    if (BeginCard("load")) {
        ImGui::SeparatorText(tr(S::LoadModel));
        ImGui::BeginDisabled(busy());
        bool go = false;
        if (has_native_dialogs()) {
            if (ImGui::Button(tr(S::Browse))) {
                std::string p = open_file_dialog(DLG_MODEL);
                if (!p.empty()) { std::snprintf(A.model_path, sizeof A.model_path, "%s", p.c_str()); go = true; }
            }
            ImGui::SameLine();
        }
        if (PrimaryButton(tr(S::Load), ImVec2(140 * A.sc, 0))) go = true;
        if (go && A.model_path[0] && !busy()) {
            std::string err;
            auto m = Model::load(A.model_path, &err);
            A.file_ok = (bool)m;
            A.file_msg = m ? tr(S::LoadedModel) : err;
            if (m) {
                A.model = m;
                A.spec = m->spec;
                reset_generation();
                A.feat_in.clear();
                if (model_matches_data() && A.data.kind == DATA_TEXT) A.data.encode_with(m->vocab);
            }
        }
        ImGui::EndDisabled();
        if (busy()) Muted(tr(S::Busy));
    }
    EndCard();

    if (!A.file_msg.empty()) {
        if (A.file_ok) ImGui::TextColored(kGood, "%s", A.file_msg.c_str());
        else {
            char b[512];
            std::snprintf(b, sizeof b, tr(S::Failed), A.file_msg.c_str());
            ImGui::TextColored(kBad, "%s", b);
        }
    }
}

void apply_theme(float sc) {
    ImGuiStyle& st = ImGui::GetStyle();
    st = ImGuiStyle();
    ImGui::StyleColorsDark();
    st.WindowRounding = 0; st.ChildRounding = 10; st.FrameRounding = 6; st.GrabRounding = 6; st.PopupRounding = 6;
    st.WindowPadding = ImVec2(22, 18); st.FramePadding = ImVec2(10, 6); st.ItemSpacing = ImVec2(10, 8);
    st.WindowBorderSize = 0; st.ChildBorderSize = 1; st.ScrollbarSize = 13;
    st.ScaleAllSizes(sc);
    ImVec4* c = st.Colors;
    c[ImGuiCol_Text] = ImVec4(0.95f, 0.93f, 0.90f, 1);
    c[ImGuiCol_WindowBg] = ImVec4(0.086f, 0.074f, 0.066f, 1);
    c[ImGuiCol_ChildBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_PopupBg] = ImVec4(0.12f, 0.105f, 0.095f, 1);
    c[ImGuiCol_Border] = ImVec4(0.25f, 0.22f, 0.20f, 1);
    c[ImGuiCol_FrameBg] = ImVec4(0.17f, 0.15f, 0.135f, 1);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.23f, 0.20f, 0.18f, 1);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.28f, 0.23f, 0.20f, 1);
    c[ImGuiCol_Button] = ImVec4(0.22f, 0.19f, 0.17f, 1);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.32f, 0.26f, 0.22f, 1);
    c[ImGuiCol_ButtonActive] = ImVec4(0.40f, 0.30f, 0.22f, 1);
    c[ImGuiCol_Header] = ImVec4(0.90f, 0.40f, 0.06f, 0.35f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.90f, 0.40f, 0.06f, 0.50f);
    c[ImGuiCol_HeaderActive] = ImVec4(0.90f, 0.40f, 0.06f, 0.70f);
    c[ImGuiCol_CheckMark] = kEmber;
    c[ImGuiCol_SliderGrab] = kEmber;
    c[ImGuiCol_SliderGrabActive] = kEmberHi;
    c[ImGuiCol_Separator] = ImVec4(0.25f, 0.22f, 0.20f, 1);
    c[ImGuiCol_TableHeaderBg] = ImVec4(0.17f, 0.15f, 0.135f, 1);
    c[ImGuiCol_TableBorderStrong] = c[ImGuiCol_TableBorderLight] = ImVec4(0.25f, 0.22f, 0.20f, 1);
    c[ImGuiCol_TableRowBgAlt] = ImVec4(1, 1, 1, 0.025f);
    c[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_TextSelectedBg] = ImVec4(0.90f, 0.40f, 0.06f, 0.45f);
}

}  // namespace

// -------------------------------------------------------------------- API

void init(float ui_scale, ImFont* title_font) {
    A.sc = ui_scale;
    A.title_font = title_font;
    A.sys = detect_system();
    A.cfg.threads = recommended_threads(A.sys);
    int lang = system_language();
    std::ifstream in(upath(settings_file()));
    std::string key;
    int value;
    while (in >> key >> value)
        if (key == "lang") lang = value;
    set_language(lang);
    apply_theme(ui_scale);
}

void shutdown() {
    A.trainer.stop();
    std::ofstream out(upath(settings_file()));
    out << "lang " << get_language() << "\n";
}

void set_page(int page) {
    if (page >= 0 && page < PAGE_COUNT) A.page = page;
}

bool load_data(const char* path) {
    std::string p = path;
    std::string ext = p.size() > 4 ? p.substr(p.size() - 4) : "";
    return load_data_impl(p, ext == ".csv" || ext == ".tsv" ? 1 : 0, -1);
}

void start_training() { begin_training(); }

void frame() {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    float sbw = 232 * A.sc;
    ImGuiWindowFlags wf = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                          ImGuiWindowFlags_NoBringToFrontOnFocus;

    // ---- sidebar
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(ImVec2(sbw, vp->WorkSize.y));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.066f, 0.056f, 0.050f, 1));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14 * A.sc, 16 * A.sc));
    ImGui::Begin("##sidebar", nullptr, wf);
    {
        ImVec2 p = ImGui::GetCursorScreenPos();
        float sz = 44 * A.sc;
        draw_logo(ImGui::GetWindowDrawList(), p, sz);
        ImGui::SetCursorScreenPos(ImVec2(p.x + sz + 12 * A.sc, p.y + 2 * A.sc));
        ImGui::BeginGroup();
        ImGui::TextUnformatted("Kor Studio");
        ImGui::TextColored(kMuted, "v0.1.0");
        ImGui::EndGroup();
        ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + sz + 20 * A.sc));

        static const S nav[PAGE_COUNT] = {S::NavHome, S::NavData, S::NavModel, S::NavTrain, S::NavTest, S::NavFiles};
        ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, ImVec2(0.f, 0.5f));
        for (int i = 0; i < PAGE_COUNT; ++i) {
            ImGui::PushID(i);
            ImVec2 q = ImGui::GetCursorScreenPos();
            std::string label = std::string("        ") + tr(nav[i]);
            if (ImGui::Selectable(label.c_str(), A.page == i, 0, ImVec2(0, 34 * A.sc))) A.page = i;
            ImU32 dot = A.page == i ? ImGui::GetColorU32(kEmber) : IM_COL32(90, 80, 72, 255);
            ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(q.x + 12 * A.sc, q.y + 17 * A.sc), 4 * A.sc, dot);
            ImGui::PopID();
        }
        ImGui::PopStyleVar();

        if (busy()) {
            TrainStats st = A.trainer.stats();
            ImGui::Dummy(ImVec2(0, 10 * A.sc));
            ImGui::TextColored(kEmberHi, "%s", tr(A.trainer.paused() ? S::StPaused : S::StRunning));
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, kEmber);
            ImGui::ProgressBar(st.total > 0 ? (float)st.step / st.total : 0.f, ImVec2(-1, 8 * A.sc), "");
            ImGui::PopStyleColor();
        }

        ImGui::SetCursorPosY(ImGui::GetWindowHeight() - 76 * A.sc);
        ImGui::TextColored(kMuted, "%s", tr(S::Language));
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##lang", language_name(get_language()))) {
            for (int l = 0; l < LANG_COUNT; ++l)
                if (ImGui::Selectable(language_name(l), l == get_language())) set_language(l);
            ImGui::EndCombo();
        }
    }
    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();

    // ---- content
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + sbw, vp->WorkPos.y));
    ImGui::SetNextWindowSize(ImVec2(vp->WorkSize.x - sbw, vp->WorkSize.y));
    ImGui::Begin("##content", nullptr, wf);
    switch (A.page) {
        case PAGE_HOME: page_home(); break;
        case PAGE_DATA: page_data(); break;
        case PAGE_MODEL: page_model(); break;
        case PAGE_TRAIN: page_train(); break;
        case PAGE_TEST: page_test(); break;
        case PAGE_FILES: page_files(); break;
    }
    if (A.page != PAGE_TEST) pump_generation();
    ImGui::End();
}

}  // namespace app
