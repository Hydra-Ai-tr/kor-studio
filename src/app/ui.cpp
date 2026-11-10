#include "ui.h"

#include "i18n.h"
#include "platform.h"
#include "plugins.h"

#include "../core/kernels.h"
#include "../core/sysinfo.h"

#include "imgui.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifndef KOR_VERSION
#define KOR_VERSION "dev"
#endif

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

    // training log (also written to a file in <config>/logs)
    enum Tone { T_NORMAL = 0, T_GOOD, T_BAD, T_ACCENT, T_MUTED };
    struct LogLine { std::string text; int tone; };
    std::vector<LogLine> log;
    std::ofstream log_file;
    std::string log_path;
    int want_tab = -1;
    bool fold_settings = false;

    // question tests ("question => answer" per line)
    std::string q_tag, a_tag;  // dialogue tags of the loaded text, empty for plain text
    char tests[32768] = "";
    std::vector<Probe> run_tests;  // the tests the results below belong to
    std::vector<ProbeResult> results;
    int results_step = 0;          // 0 = manual run
    bool stop_on_pass = false;

    // plugins
    std::vector<Plugin> plugins;
    std::vector<std::array<char, 512>> plugin_args;
    bool plugins_scanned = false;
    PythonFinder python;
    PluginRunner runner;
    int plugin_running = -1;
    std::string plugin_out_dir, plugin_msg;
    bool plugin_ok = true;
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

// Local time, e.g. "2026-11-10 22:05:31" (or "2026-11-10_22-05-31" for file names).
std::string timestamp(bool for_file) {
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    char b[32];
    std::strftime(b, sizeof b, for_file ? "%Y-%m-%d_%H-%M-%S" : "%Y-%m-%d %H:%M:%S", &tm);
    return b;
}

void Tooltip(const char* s) {
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("%s", s);
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

// ------------------------------------------------------------ training log

std::string logs_dir() { return (upath(config_dir()) / "logs").u8string(); }

void log_line(const std::string& text, int tone = State::T_NORMAL) {
    A.log.push_back({text, tone});
    if (A.log_file) {
        A.log_file << text << "\n";
        A.log_file.flush();
    }
}

template <class... Args>
void log_fmt(int tone, S fmt, Args... args) {
    char b[2048];
    std::snprintf(b, sizeof b, tr(fmt), args...);
    log_line(b, tone);
}

void start_log() {
    namespace fs = std::filesystem;
    A.log.clear();
    A.log_file.close();
    std::error_code ec;
    fs::create_directories(upath(logs_dir()), ec);
    A.log_path = (upath(logs_dir()) / ("train-" + timestamp(true) + ".log")).u8string();
    A.log_file.open(upath(A.log_path), std::ios::binary);
    const ModelSpec& s = A.model->spec;
    log_fmt(State::T_ACCENT, S::LogHeader, KOR_VERSION, timestamp(false).c_str());
    log_fmt(State::T_MUTED, S::LogData, A.data.path.c_str());
    log_fmt(State::T_MUTED, S::LogModel, tr(s.kind == MODEL_GPT ? S::TextModel : S::TableModel),
            fmt_count((double)A.model->param_count(false)).c_str(), fmt_count((double)A.model->param_count(true)).c_str());
    log_fmt(State::T_MUTED, S::LogCfg, A.cfg.steps, A.cfg.batch, A.cfg.accum, (double)A.cfg.lr,
            A.cfg.optimizer == OPT_SGD ? "SGD" : "AdamW", A.cfg.threads);
    if (!A.cfg.probes.empty()) log_fmt(State::T_MUTED, S::LogTests, (int)A.cfg.probes.size());
}

// Moves new training events into the log; called every frame.
void pump_events() {
    std::vector<TrainEvent> ev;
    A.trainer.drain_events(ev);
    for (const TrainEvent& e : ev) {
        char b[2048];
        std::string t = "[" + fmt_time(e.time) + "] ";
        switch (e.kind) {
            case EV_START:
                std::snprintf(b, sizeof b, tr(S::LogStart), fmt_count((double)e.n1).c_str(), (int)e.n2, e.text.c_str());
                log_line(t + b);
                break;
            case EV_EVAL: {
                std::snprintf(b, sizeof b, tr(S::LogEval), e.step, e.loss, e.val_loss);
                std::string s = t + b;
                if (e.val_acc >= 0) { std::snprintf(b, sizeof b, tr(S::LogAcc), e.val_acc * 100); s += b; }
                if (e.n1) s += tr(S::LogBest);
                log_line(s, e.n1 ? State::T_GOOD : State::T_NORMAL);
                break;
            }
            case EV_PROBES: {
                std::snprintf(b, sizeof b, tr(S::LogProbes), e.step, (int)e.n1, (int)e.n2, e.text.c_str());
                log_line(t + b, State::T_ACCENT);
                TrainStats st = A.trainer.stats();
                A.results = st.probe_results;
                A.results_step = e.step;
                break;
            }
            case EV_EARLY_STOP:
                std::snprintf(b, sizeof b, tr(S::LogEarly), e.step, (int)e.n1);
                log_line(t + b, State::T_ACCENT);
                break;
            case EV_TARGET:
                std::snprintf(b, sizeof b, tr(S::LogTarget), e.step, (int)e.n1, (int)e.n2);
                log_line(t + b, State::T_GOOD);
                break;
            case EV_RESTORED:
                if (e.n2 > 0) std::snprintf(b, sizeof b, tr(S::LogRestoredTests), e.step, (int)e.n1, (int)e.n2);
                else std::snprintf(b, sizeof b, tr(S::LogRestored), e.step, e.val_loss);
                log_line(t + b, State::T_ACCENT);
                break;
            case EV_DONE:
                std::snprintf(b, sizeof b, tr(S::LogDone), e.step, fmt_time(e.time).c_str());
                log_line(t + b, State::T_GOOD);
                break;
            case EV_STOPPED:
                std::snprintf(b, sizeof b, tr(S::LogStopped), e.step);
                log_line(t + b, State::T_MUTED);
                break;
            case EV_ERROR:
                std::snprintf(b, sizeof b, tr(S::LogError), e.text.c_str());
                log_line(t + b, State::T_BAD);
                break;
        }
        if (e.kind == EV_DONE || e.kind == EV_STOPPED || e.kind == EV_ERROR) A.log_file.close();
    }
}

// --------------------------------------------------------- question tests

std::vector<Probe> current_tests() { return parse_tests(A.tests, A.q_tag, A.a_tag); }

void set_tests(const std::string& text) { std::snprintf(A.tests, sizeof A.tests, "%s", text.c_str()); }

void tests_from_data() {
    set_tests(format_tests(A.data.dialogue_pairs(12)));
    A.results.clear();
}

bool can_run_tests() { return A.model && A.model->spec.kind == MODEL_GPT && !busy(); }

void run_tests_now() {
    if (!can_run_tests()) return;
    A.run_tests = current_tests();
    A.results.clear();
    A.results_step = 0;
    int pass = 0;
    {
        std::lock_guard<std::mutex> lk(A.trainer.model_mu);
        for (auto& p : A.run_tests) {
            A.results.push_back(run_probe(*A.model, p));
            pass += A.results.back().pass;
        }
    }
    if (!A.run_tests.empty()) log_fmt(State::T_ACCENT, S::LogManual, pass, (int)A.run_tests.size());
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
    TrainConfig old = A.cfg;
    A.cfg = A.plan.cfg;
    A.cfg.keep_best = old.keep_best;
    A.cfg.patience = old.patience;
    A.cfg.probe_every = std::max(1, A.cfg.steps / 10);
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
    A.q_tag.clear();
    A.a_tag.clear();
    if (!A.data.dialogue_tags(A.q_tag, A.a_tag)) { A.q_tag.clear(); A.a_tag.clear(); }
    if (A.data.kind == DATA_TEXT) tests_from_data();
    else set_tests("");
    A.stop_on_pass = !A.q_tag.empty();
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
    A.cfg.probes = A.data.kind == DATA_TEXT ? current_tests() : std::vector<Probe>();
    A.cfg.probe_target = A.stop_on_pass ? 1.f : 0.f;
    A.run_tests = A.cfg.probes;
    A.results.clear();
    start_log();
    if (!A.trainer.start(A.model, &A.data, A.cfg)) {
        log_fmt(State::T_BAD, S::LogError, A.trainer.stats().error.c_str());
        A.log_file.close();
    } else {
        A.fold_settings = true;
    }
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

ImVec4 tone_color(int tone) {
    switch (tone) {
        case State::T_GOOD: return kGood;
        case State::T_BAD: return kBad;
        case State::T_ACCENT: return kEmberHi;
        case State::T_MUTED: return kMuted;
    }
    return ImGui::GetStyleColorVec4(ImGuiCol_Text);
}

void log_view(float height) {
    ImGui::BeginDisabled(A.log.empty());
    if (ImGui::Button(tr(S::CopyLog))) {
        std::string all;
        for (auto& l : A.log) all += l.text + "\n";
        ImGui::SetClipboardText(all.c_str());
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button(tr(S::OpenLogs))) {
        std::error_code ec;
        std::filesystem::create_directories(upath(logs_dir()), ec);
        open_folder(logs_dir());
    }
    if (!A.log_path.empty()) {
        ImGui::SameLine();
        char b[2400];
        std::snprintf(b, sizeof b, tr(S::LogFile), A.log_path.c_str());
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(kMuted, "%s", b);
    }
    ImGui::BeginChild("##log", ImVec2(0, std::max(120 * A.sc, height - ImGui::GetFrameHeightWithSpacing())), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_HorizontalScrollbar);
    if (A.log.empty()) Muted(tr(S::LogEmpty));
    bool at_end = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4;
    ImGuiListClipper clip;
    clip.Begin((int)A.log.size());
    while (clip.Step())
        for (int i = clip.DisplayStart; i < clip.DisplayEnd; ++i)
            ImGui::TextColored(tone_color(A.log[i].tone), "%s", A.log[i].text.c_str());
    if (at_end) ImGui::SetScrollHereY(1.f);
    ImGui::EndChild();
}

void tests_view(const TrainStats& st) {
    char b[1024];
    Muted(tr(S::TestsHelp));
    if (!A.q_tag.empty()) {
        std::snprintf(b, sizeof b, tr(S::DialogDetected), A.q_tag.c_str(), A.a_tag.c_str());
        ImGui::TextColored(kBlue, "%s", b);
    } else {
        Muted(tr(S::NoDialog));
    }
    bool run = busy();
    ImGui::BeginDisabled(run);
    ImGui::InputTextMultiline("##tests", A.tests, sizeof A.tests, ImVec2(-1, 120 * A.sc));
    ImGui::EndDisabled();
    std::vector<Probe> tests = current_tests();
    std::snprintf(b, sizeof b, tr(S::TestsCount), (int)tests.size());
    ImGui::TextColored(kMuted, "%s", b);
    if (!A.q_tag.empty()) {
        ImGui::SameLine();
        ImGui::BeginDisabled(run);
        if (ImGui::SmallButton(tr(S::UseDataQuestions))) tests_from_data();
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(!can_run_tests() || tests.empty());
    if (ImGui::SmallButton(tr(S::RunTestsNow))) run_tests_now();
    ImGui::EndDisabled();
    if (tests.empty()) Muted(tr(S::NoTests));
    else if (!A.model || A.model->spec.kind != MODEL_GPT) Muted(tr(S::NeedTextModel));

    if (A.results.empty()) return;
    int pass = 0;
    for (auto& r : A.results) pass += r.pass;
    ImGui::Dummy(ImVec2(0, 2 * A.sc));
    std::snprintf(b, sizeof b, tr(S::TestsPassed), pass, (int)A.results.size());
    if (A.title_font) ImGui::PushFont(A.title_font);
    ImGui::TextColored(pass == (int)A.results.size() ? kGood : pass > 0 ? kEmberHi : kBad, "%s", b);
    if (A.title_font) ImGui::PopFont();
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    if (A.results_step > 0) std::snprintf(b, sizeof b, tr(S::ResultsAt), A.results_step);
    else std::snprintf(b, sizeof b, "%s", tr(S::ResultsManual));
    ImGui::TextColored(kMuted, "%s", b);

    if (st.probe_hist.size() >= 2 && A.results_step > 0) {
        std::vector<float> h(st.probe_hist.begin(), st.probe_hist.end());
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, kEmber);
        ImGui::PlotHistogram("##phist", h.data(), (int)h.size(), 0, tr(S::TestsHistory), 0.f, (float)std::max(1, st.probe_total),
                             ImVec2(-1, 60 * A.sc));
        ImGui::PopStyleColor();
    }

    int n = (int)std::min(A.results.size(), A.run_tests.size());
    ImGuiTableFlags tf = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable;
    float th = std::max(140 * A.sc, ImGui::GetContentRegionAvail().y - 4 * A.sc);
    if (ImGui::BeginTable("##results", 4, tf, ImVec2(0, th))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn(tr(S::ColQuestion), ImGuiTableColumnFlags_WidthStretch, 1.2f);
        ImGui::TableSetupColumn(tr(S::ColExpected), ImGuiTableColumnFlags_WidthStretch, 1.f);
        ImGui::TableSetupColumn(tr(S::ColAnswer), ImGuiTableColumnFlags_WidthStretch, 1.f);
        ImGui::TableSetupColumn(tr(S::Result), ImGuiTableColumnFlags_WidthFixed, 70 * A.sc);
        ImGui::TableHeadersRow();
        for (int i = 0; i < n; ++i) {
            const ProbeResult& r = A.results[i];
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextWrapped("%s", A.run_tests[i].question.c_str());
            ImGui::TableNextColumn();
            ImGui::TextWrapped("%s", A.run_tests[i].expected.c_str());
            ImGui::TableNextColumn();
            ImGui::PushStyleColor(ImGuiCol_Text, r.pass ? kGood : kBad);
            ImGui::TextWrapped("%s", r.answer.empty() ? "-" : r.answer.c_str());
            ImGui::PopStyleColor();
            ImGui::TableNextColumn();
            ImGui::TextColored(r.pass ? kGood : kBad, "%s", tr(r.pass ? S::Pass : S::Fail));
        }
        ImGui::EndTable();
    }
}

void train_settings(bool run) {
    ImGui::BeginDisabled(run);
    float w = (ImGui::GetContentRegionAvail().x - 24 * A.sc) / 2;
    ImGui::BeginGroup();
    ImGui::PushItemWidth(170 * A.sc);
    ImGui::InputInt(tr(S::Steps), &A.cfg.steps, 100, 1000);
    ImGui::InputInt(tr(S::BatchSize), &A.cfg.batch, 1, 8);
    ImGui::InputInt(tr(S::Accum), &A.cfg.accum, 1, 2);
    ImGui::InputInt(tr(S::EvalEvery), &A.cfg.eval_every, 10, 100);
    ImGui::InputInt(tr(S::Patience), &A.cfg.patience, 1, 5);
    Tooltip(tr(S::PatienceHelp));
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
    if (A.data.kind == DATA_TEXT) {
        ImGui::InputInt(tr(S::TestEvery), &A.cfg.probe_every, 50, 500);
        Tooltip(tr(S::TestEveryHelp));
    }
    ImGui::PopItemWidth();
    ImGui::EndGroup();
    ImGui::Checkbox(tr(S::KeepBest), &A.cfg.keep_best);
    Tooltip(tr(S::KeepBestHelp));
    if (A.data.kind == DATA_TEXT) {
        ImGui::SameLine(w + 24 * A.sc);
        ImGui::Checkbox(tr(S::StopOnPass), &A.stop_on_pass);
    }
    A.cfg.patience = std::max(0, std::min(1000, A.cfg.patience));
    A.cfg.probe_every = std::max(0, A.cfg.probe_every);
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
    char eff[96], b[256];
    std::snprintf(eff, sizeof eff, tr(S::EffBatch), A.cfg.batch * A.cfg.accum);
    std::snprintf(b, sizeof b, "%s   |   %s: %s", eff, tr(S::EstRam), fmt_mb(mb).c_str());
    ImGui::TextColored(kMuted, "%s", b);
}

void page_train() {
    Title(tr(S::NavTrain));
    if (A.data.kind == DATA_NONE) { need_data_notice(); return; }
    TrainStats st = A.trainer.stats();
    bool run = busy();
    char b[256];

    if (BeginCard("cfg")) {
        // Folded while training so the chart, log and tests get the room.
        if (A.fold_settings) {
            ImGui::SetNextItemOpen(false);
            A.fold_settings = false;
        }
        if (ImGui::CollapsingHeader(tr(S::Settings), ImGuiTreeNodeFlags_DefaultOpen)) train_settings(run);
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

    if (ImGui::BeginTabBar("##traintabs")) {
        auto flags = [](int tab) { return A.want_tab == tab ? ImGuiTabItemFlags_SetSelected : 0; };
        float h = std::max(220 * A.sc, ImGui::GetContentRegionAvail().y - 40 * A.sc);
        if (ImGui::BeginTabItem(tr(S::TabChart), nullptr, flags(0))) {
            plot_loss(st, std::max(180 * A.sc, ImGui::GetContentRegionAvail().y - 8 * A.sc));
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(tr(S::TabLog), nullptr, flags(1))) {
            log_view(h);
            ImGui::EndTabItem();
        }
        if (A.data.kind == DATA_TEXT && ImGui::BeginTabItem(tr(S::TabTests), nullptr, flags(2))) {
            tests_view(st);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    A.want_tab = -1;
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

// ---------------------------------------------------------------- plugins

std::string user_plugins_dir() { return (upath(config_dir()) / "plugins").u8string(); }

// The bundled plugins next to the program (or in the source tree), then the user's own folder.
std::vector<std::string> plugin_dirs() {
    namespace fs = std::filesystem;
    std::vector<std::string> dirs;
    fs::path exe = upath(exe_dir());
    for (const fs::path& base : {exe / "plugins", exe / ".." / "plugins", exe / ".." / ".." / "plugins"}) {
        std::error_code ec;
        if (fs::is_directory(base, ec)) {
            dirs.push_back(fs::weakly_canonical(base, ec).u8string());
            break;
        }
    }
    dirs.push_back(user_plugins_dir());
    return dirs;
}

void scan_all_plugins() {
    A.plugins = scan_plugins(plugin_dirs());
    A.plugin_args.assign(A.plugins.size(), {});
    for (size_t i = 0; i < A.plugins.size(); ++i)
        std::snprintf(A.plugin_args[i].data(), A.plugin_args[i].size(), "%s", A.plugins[i].args.c_str());
    A.plugins_scanned = true;
}

bool read_file_text(const std::filesystem::path& p, std::string& out) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

void run_plugin(int i) {
    namespace fs = std::filesystem;
    if (busy() || A.runner.running() || !A.python.found() || i < 0 || i >= (int)A.plugins.size()) return;
    const Plugin& p = A.plugins[i];
    fs::path out = upath(config_dir()) / "plugin-runs" / (upath(p.path).stem().u8string() + "-" + timestamp(true));
    std::error_code ec;
    fs::create_directories(out, ec);
    A.plugin_out_dir = out.u8string();

    // Inputs: the current dataset, model and question tests.
    std::string model_file;
    if (A.model) {
        std::string err;
        std::lock_guard<std::mutex> lk(A.trainer.model_mu);
        if (A.model->save((out / "input-model.kor").u8string(), false, &err)) model_file = (out / "input-model.kor").u8string();
    }
    std::string tests_file = (out / "input-tests.txt").u8string();
    { std::ofstream t(upath(tests_file), std::ios::binary); t << A.tests; }
#if defined(_WIN32)
    const char* cli_name = "kor-cli.exe";
    const char sep = ';';
#else
    const char* cli_name = "kor-cli";
    const char sep = ':';
#endif
    fs::path cli = upath(exe_dir()) / cli_name;
    std::string pypath;
    for (auto& d : plugin_dirs()) pypath += (pypath.empty() ? "" : std::string(1, sep)) + d;
    static const char* const lang_codes[LANG_COUNT] = {"en", "tr", "ru"};
    std::vector<std::pair<std::string, std::string>> env = {
        {"KOR_OUT", A.plugin_out_dir},
        {"KOR_DATA", A.data.kind != DATA_NONE ? A.data.path : ""},
        {"KOR_MODEL", model_file},
        {"KOR_TESTS", tests_file},
        {"KOR_CLI", fs::exists(cli, ec) ? cli.u8string() : ""},
        {"KOR_LANG", lang_codes[get_language()]},
        {"PYTHONPATH", pypath},
        {"PYTHONIOENCODING", "utf-8"},
        {"PYTHONUNBUFFERED", "1"},
    };
    std::vector<std::string> cmd = A.python.command();
    cmd.push_back("-X");
    cmd.push_back("utf8");
    cmd.push_back(p.path);
    for (auto& a : split_args(A.plugin_args[i].data())) cmd.push_back(a);
    if (A.runner.start(cmd, env)) {
        A.plugin_running = i;
        char b[512];
        std::snprintf(b, sizeof b, tr(S::PluginRunning), p.title(get_language()).c_str());
        A.plugin_msg = b;
        A.plugin_ok = true;
    }
}

// Picks up a finished plugin run and loads what it produced; called every frame.
void poll_plugin() {
    namespace fs = std::filesystem;
    int code = 0;
    if (!A.runner.take_finished(code)) return;
    int i = A.plugin_running;
    A.plugin_running = -1;
    std::string name = i >= 0 && i < (int)A.plugins.size() ? A.plugins[i].title(get_language()) : "?";
    char b[2400];
    A.plugin_ok = code == 0;
    if (code == -2) std::snprintf(b, sizeof b, tr(S::PluginCancelled), name.c_str());
    else if (code == -1) std::snprintf(b, sizeof b, "%s", tr(S::PluginNoStart));
    else if (code != 0) std::snprintf(b, sizeof b, tr(S::PluginFailed), name.c_str(), code);
    else std::snprintf(b, sizeof b, tr(S::PluginDone), name.c_str());
    A.plugin_msg = b;
    if (code != 0) return;

    fs::path out = upath(A.plugin_out_dir);
    std::error_code ec;
    for (const char* f : {"data.txt", "data.csv", "data.tsv"}) {
        fs::path dp = out / f;
        if (!fs::exists(dp, ec)) continue;
        int kind = std::strcmp(f, "data.txt") == 0 ? 0 : 1;
        if (load_data_impl(dp.u8string(), kind, -1)) {
            std::snprintf(b, sizeof b, tr(S::PluginData), dp.u8string().c_str());
            A.plugin_msg += std::string("\n") + b;
        } else {
            std::snprintf(b, sizeof b, tr(S::Failed), A.data_msg.c_str());
            A.plugin_msg += std::string("\n") + b;
            A.plugin_ok = false;
        }
        break;
    }
    std::string tests;
    if (read_file_text(out / "tests.txt", tests)) {
        set_tests(tests);
        A.results.clear();
        std::snprintf(b, sizeof b, tr(S::PluginTests), (int)current_tests().size());
        A.plugin_msg += std::string("\n") + b;
    }
}

void page_plugins() {
    Title(tr(S::NavPlugins));
    Muted(tr(S::PluginsHelp));
    ImGui::Dummy(ImVec2(0, 4 * A.sc));
    if (!A.plugins_scanned) {
        scan_all_plugins();
        A.python.start();
    }
    char b[2400];

    if (BeginCard("python")) {
        ImGui::AlignTextToFramePadding();
        if (A.python.searching()) {
            ImGui::TextColored(kMuted, "%s", tr(S::PythonChecking));
        } else if (A.python.found()) {
            std::snprintf(b, sizeof b, tr(S::PythonFound), A.python.version().c_str());
            ImGui::TextColored(kGood, "%s", b);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, kBad);
            ImGui::TextWrapped("%s", tr(S::PythonMissing));
            ImGui::PopStyleColor();
        }
        if (!A.python.searching() && !A.python.found()) {
            if (ImGui::Button(tr(S::CheckAgain))) A.python.start();
        }
    }
    EndCard();

    if (ImGui::Button(tr(S::OpenPlugins))) {
        std::error_code ec;
        std::filesystem::create_directories(upath(user_plugins_dir()), ec);
        open_folder(user_plugins_dir());
    }
    ImGui::SameLine();
    if (ImGui::Button(tr(S::RefreshList))) scan_all_plugins();
    ImGui::TextColored(kMuted, "%s", tr(S::PluginDirs));
    for (auto& d : plugin_dirs()) ImGui::TextColored(kMuted, "    %s", d.c_str());
    ImGui::Dummy(ImVec2(0, 4 * A.sc));

    if (A.plugins.empty()) Muted(tr(S::NoPlugins));
    bool can_run = A.python.found() && !busy() && !A.runner.running();
    if (busy()) ImGui::TextColored(kEmberHi, "%s", tr(S::PluginBusy));
    int lang = get_language();
    for (int i = 0; i < (int)A.plugins.size(); ++i) {
        const Plugin& p = A.plugins[i];
        ImGui::PushID(i);
        if (BeginCard("plugin")) {
            ImGui::TextColored(kEmberHi, "%s", p.title(lang).c_str());
            ImGui::SameLine();
            ImGui::TextColored(kMuted, "%s", p.file.c_str());
            if (!p.description(lang).empty()) {
                ImGui::PushTextWrapPos(0);
                ImGui::TextWrapped("%s", p.description(lang).c_str());
                ImGui::PopTextWrapPos();
            }
            ImGui::SetNextItemWidth(-200 * A.sc);
            ImGui::InputTextWithHint("##args", tr(S::Arguments), A.plugin_args[i].data(), A.plugin_args[i].size());
            ImGui::SameLine();
            if (A.plugin_running == i && A.runner.running()) {
                if (ImGui::Button(tr(S::Stop), ImVec2(120 * A.sc, 0))) A.runner.cancel();
            } else {
                ImGui::BeginDisabled(!can_run);
                if (PrimaryButton(tr(S::Run), ImVec2(120 * A.sc, 0))) run_plugin(i);
                ImGui::EndDisabled();
            }
        }
        EndCard();
        ImGui::PopID();
    }

    if (!A.plugin_msg.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, A.runner.running() ? kEmberHi : A.plugin_ok ? kGood : kBad);
        ImGui::TextWrapped("%s", A.plugin_msg.c_str());
        ImGui::PopStyleColor();
    }
    std::string out = A.runner.output();
    if (out.empty()) return;
    ImGui::SeparatorText(tr(S::PluginOutput));
    if (ImGui::SmallButton(tr(S::Clear))) A.runner.clear_output();
    ImGui::BeginChild("##pout", ImVec2(0, std::max(160 * A.sc, ImGui::GetContentRegionAvail().y)), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_HorizontalScrollbar);
    bool at_end = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4;
    ImGui::TextUnformatted(out.data(), out.data() + out.size());
    if (at_end) ImGui::SetScrollHereY(1.f);
    ImGui::EndChild();
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
    A.runner.cancel();
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

void set_train_tab(int tab) { A.want_tab = tab; }

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
        ImGui::TextColored(kMuted, "v%s", KOR_VERSION);
        ImGui::EndGroup();
        ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + sz + 20 * A.sc));

        static const S nav[] = {S::NavHome, S::NavData, S::NavModel, S::NavTrain, S::NavTest, S::NavFiles, S::NavPlugins};
        static_assert(sizeof(nav) / sizeof(nav[0]) == PAGE_COUNT, "one label per page");
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
        case PAGE_PLUGINS: page_plugins(); break;
    }
    pump_events();
    poll_plugin();
    if (A.page != PAGE_TEST) pump_generation();
    ImGui::End();
}

}  // namespace app
