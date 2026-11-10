#include "ui.h"

#include "i18n.h"
#include "kit.h"
#include "platform.h"
#include "plugins.h"

#include "../core/kernels.h"
#include "../core/media.h"
#include "../core/sysinfo.h"
#include "../skills/json.h"
#include "../skills/skill.h"

#include "imgui.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifndef KOR_VERSION
#define KOR_VERSION "dev"
#endif

using namespace kor;
using namespace kit;

namespace app {
namespace {

// ------------------------------------------------------------------ state

struct State {
    int page = PAGE_HOME;

    SysInfo sys;
    Dataset data;
    Dataset train_data;  // data without the test questions; read by the trainer thread
    std::shared_ptr<Model> model;
    Trainer trainer;
    ModelSpec spec;
    TrainConfig cfg;
    AutoPlan plan;
    bool has_plan = false;

    char data_path[2048] = "";
    int data_kind = 0;  // 0 text, 1 table, 2 images, 3 sound
    std::string data_msg;
    bool data_ok = true;
    bool subword = false;
    int subword_size = 1000;
    int img_size = 16, img_colors = 32;
    bool img_mirror = true;
    int snd_rate = 8000;
    std::vector<ImTextureID> data_tex;  // preview of the loaded images

    // pictures and sound made in the playground (on a worker thread)
    struct MediaGen {
        std::thread th;
        std::atomic<bool> running{false}, cancel{false};
        std::atomic<float> progress{0.f};
        std::mutex mu;  // guards images and sound
        std::vector<Image> images;
        Sound sound;
    } mg;
    int gen_count = 9;
    float gen_seconds = 3.f;
    std::vector<ImTextureID> gen_tex;
    std::string media_msg;
    bool media_ok = true;

    int add_type = L_DENSE;
    int lora_rank = 8;
    // Fine-tuning: the model came from a file; characters added to it / left
    // out of the data because LoRA is on.
    bool from_file = false;
    int ft_added = 0, ft_dropped = 0;

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
    int train_tab = 0;
    bool settings_open = true;

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

    // skills and sharing
    std::set<std::string> skills_on;                // turned-on skills (file names)
    std::map<std::string, std::string> saved_args;  // plugin arguments kept in the settings file
    std::vector<std::string> active_skills;         // skills of the current/last run
    enum RunKind { RUN_TOOL = 0, RUN_INSTALL, RUN_INDEX, RUN_NEW };
    int run_kind = RUN_TOOL;
    char install_src[1024] = "";
    struct CommunityItem {
        std::string id, type, author, url, name[3], desc[3];
    };
    std::vector<CommunityItem> community;
    int community_state = 0;  // 0 not loaded, 1 loading, 2 loaded, 3 failed
    bool save_settings = true;
    bool scroll_end = false, want_community = false;
};
State A;

bool busy() { return A.trainer.running(); }

// ---------------------------------------------------------------- helpers

template <class... Args>
std::string strf(const char* fmt, Args... args) {
    char b[2048];
    std::snprintf(b, sizeof b, fmt, args...);
    return b;
}

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

// Local time, e.g. "2026-11-10 22:05:31" (or "2026-11-10_22-05-31" for file names).
std::string timestamp(bool for_file) {
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    char b[32];
    std::strftime(b, sizeof b, for_file ? "%Y-%m-%d_%H-%M-%S" : "%Y-%m-%d %H:%M:%S", &tm);
    return b;
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

// Places the cursor so that `h` pixels of content end at the bottom of a card
// that is min_h tall. Returns the natural card height (without the push).
float push_to_bottom(float min_h, float h) {
    float y = ImGui::GetCursorPosY();
    float target = min_h - card_padding_y() - h;
    if (target > y) ImGui::SetCursorPosY(target);
    return y + h + card_padding_y();
}

// Inset box (logs, previews, consoles) filling `height`.
bool begin_inset(const char* id, float height, bool hscroll = false) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, C::inset);
    ImGui::PushStyleColor(ImGuiCol_Border, C::border);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(px(14), px(12)));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, px(10));
    bool open = ImGui::BeginChild(id, ImVec2(0, height), ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding,
                                  hscroll ? ImGuiWindowFlags_HorizontalScrollbar : 0);
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
    return open;
}

void table_headers() {
    FontScope f(F.small_bold);
    ImGui::PushStyleColor(ImGuiCol_Text, C::muted);
    ImGui::TableHeadersRow();
    ImGui::PopStyleColor();
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
    log_fmt(State::T_MUTED, S::LogModel,
            tr(s.kind != MODEL_GPT ? S::TableModel : s.media == MEDIA_IMAGE ? S::ImageModel : s.media == MEDIA_AUDIO ? S::SoundModel : S::TextModel),
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
            case EV_HOOK_LOG: {
                static const int tones[] = {State::T_NORMAL, State::T_GOOD, State::T_ACCENT, State::T_BAD, State::T_MUTED};
                log_line(t + "[" + e.source + "] " + e.text, tones[e.n1 >= 0 && e.n1 < 5 ? e.n1 : 0]);
                break;
            }
            case EV_HOOK_STOP:
                std::snprintf(b, sizeof b, tr(S::LogHookStop), e.text.empty() ? "-" : e.text.c_str());
                log_line(t + "[" + e.source + "] " + b, State::T_ACCENT);
                break;
            case EV_HOOK_LR:
                std::snprintf(b, sizeof b, tr(e.n1 ? S::LogHookLrFixed : S::LogHookLr), (double)e.loss);
                log_line(t + "[" + e.source + "] " + b, State::T_NORMAL);
                break;
            case EV_HOOK_DATA:
                if (e.n1) std::snprintf(b, sizeof b, tr(S::LogHookData), e.n1, e.n2);
                else std::snprintf(b, sizeof b, "%s", tr(S::LogHookCleared));
                log_line(t + "[" + e.source + "] " + b, State::T_NORMAL);
                break;
            case EV_HOOK_BEST:
                if (e.n1) std::snprintf(b, sizeof b, "%s", tr(S::LogHookSaved));
                else std::snprintf(b, sizeof b, tr(S::LogHookRestored), e.n2);
                log_line(t + "[" + e.source + "] " + b, State::T_ACCENT);
                break;
            case EV_HOOK_SAMPLING:
                std::snprintf(b, sizeof b, tr(S::LogHookSampling), (double)e.loss * 100, (double)e.val_loss);
                log_line(t + "[" + e.source + "] " + b, State::T_ACCENT);
                break;
            case EV_HOOK_DROPOUT:
                std::snprintf(b, sizeof b, tr(S::LogHookDropout), (double)e.loss);
                log_line(t + "[" + e.source + "] " + b, State::T_ACCENT);
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

bool can_run_tests() { return A.model && A.model->spec.kind == MODEL_GPT && A.model->spec.media == MEDIA_TEXT && !busy(); }

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
        A.spec.media = A.data.media;
        A.spec.img_size = A.data.img_size;
        A.spec.rate = A.data.rate;
        if (A.data.media == MEDIA_IMAGE) A.spec.ctx = A.data.img_size * A.data.img_size + 1;
        else A.spec.ctx = std::max(8, std::min(A.spec.ctx, (int)A.data.n_train - 2));
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

ImTextureID make_texture(const Image& img) {
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.w, img.h, 0, GL_RGB, GL_UNSIGNED_BYTE, img.rgb.data());
    return (ImTextureID)t;
}

void free_textures(std::vector<ImTextureID>& v) {
    for (ImTextureID t : v) {
        GLuint id = (GLuint)t;
        glDeleteTextures(1, &id);
    }
    v.clear();
}

// Pictures in rows that fill the available width.
void image_grid(const std::vector<ImTextureID>& tex, float cell) {
    float g = px(8), aw = ImGui::GetContentRegionAvail().x;
    int cols = std::max(1, (int)((aw + g) / (cell + g)));
    for (size_t i = 0; i < tex.size(); ++i) {
        if (i % cols) ImGui::SameLine(0, g);
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImGui::Image(tex[i], ImVec2(cell, cell));
        ImGui::GetWindowDrawList()->AddRect(p, ImVec2(p.x + cell, p.y + cell), col(C::border), px(2));
    }
}

// Sound as a min/max envelope per pixel column.
void waveform(const std::vector<float>& x, ImVec2 size) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Dummy(size);
    dl->AddRectFilled(p, ImVec2(p.x + size.x, p.y + size.y), col(C::surface3), px(6));
    float mid = p.y + size.y * 0.5f, half = size.y * 0.45f;
    dl->AddLine(ImVec2(p.x, mid), ImVec2(p.x + size.x, mid), col(C::border));
    int w = std::max(1, (int)size.x);
    if (x.empty()) return;
    for (int c = 0; c < w; ++c) {
        size_t a = x.size() * c / w, b = std::max(a + 1, x.size() * (c + 1) / w);
        float lo = 0, hi = 0;
        for (size_t i = a; i < b && i < x.size(); ++i) { lo = std::min(lo, x[i]); hi = std::max(hi, x[i]); }
        dl->AddLine(ImVec2(p.x + c, mid - hi * half), ImVec2(p.x + c, mid - lo * half + 1), col(C::accent));
    }
}

// Stops the picture/sound worker and waits for it.
void stop_media_gen() {
    A.mg.cancel = true;
    if (A.mg.th.joinable()) A.mg.th.join();
    A.mg.running = false;
}

void reset_generation() {
    A.gen_left = 0;
    A.gen_ids.clear();
    A.gen_out.clear();
    A.pred.clear();
    stop_media_gen();
    free_textures(A.gen_tex);
    A.mg.images.clear();
    A.mg.sound = Sound();
    A.media_msg.clear();
}

void build_model() {
    if (A.data.kind == DATA_NONE) return;
    finalize_spec();
    A.model = std::make_shared<Model>(A.spec, (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count());
    A.from_file = false;
    A.ft_added = A.ft_dropped = 0;
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
    if (s.kind == MODEL_GPT)
        return A.data.kind == DATA_TEXT && A.data.media == s.media &&
               (s.media != MEDIA_IMAGE || (A.data.img_size == s.img_size && A.data.vocab == A.model->vocab)) &&
               (s.media != MEDIA_AUDIO || A.data.rate == s.rate);
    return A.data.kind == DATA_TABLE && s.in_dim == A.data.n_feat && s.regression == A.data.regression &&
           (s.regression || s.out_dim == (int)A.data.labels.size());
}

// Fine-tuning a text model: characters of the data the model does not know
// yet are added to it (unless LoRA is on), then the text is encoded with the
// model's vocabulary.
void fit_data_to_model() {
    A.ft_added = 0;
    A.ft_dropped = 0;
    if (!model_matches_data() || A.data.kind != DATA_TEXT || A.data.media != MEDIA_TEXT) return;
    std::lock_guard<std::mutex> lk(A.trainer.model_mu);
    A.ft_added = A.model->extend_vocab(A.data.vocab);
    for (auto& c : A.data.vocab)
        if (std::find(A.model->vocab.begin(), A.model->vocab.end(), c) == A.model->vocab.end()) ++A.ft_dropped;
    A.data.encode_with(A.model->vocab);
}

// Reads `path` as data of `kind` (0 text, 1 table, 2 images, 3 sound) with
// the settings of the data page. Images and sound for a model loaded from a
// file are read in that model's format, so it can be fine-tuned.
bool read_data(Dataset& d, const std::string& path, int kind, int target_col, std::string& err) {
    if (kind == 1) return d.load_table(path, target_col, &err);
    const ModelSpec* ft = A.model && A.from_file && A.model->spec.kind == MODEL_GPT ? &A.model->spec : nullptr;
    if (kind == 2) {
        std::error_code ec;
        std::string dir = std::filesystem::is_regular_file(upath(path), ec) ? upath(path).parent_path().u8string() : path;
        if (ft && ft->media == MEDIA_IMAGE) return d.load_images(dir, ft->img_size, 0, A.img_mirror, &err, &A.model->vocab);
        return d.load_images(dir, A.img_size, A.img_colors, A.img_mirror, &err);
    }
    if (kind == 3) return d.load_audio(path, ft && ft->media == MEDIA_AUDIO ? ft->rate : A.snd_rate, &err);
    return d.load_text(path, &err, A.subword ? A.subword_size : 0);
}

bool load_data_impl(const std::string& path, int kind, int target_col) {
    if (busy()) return false;
    std::string err;
    Dataset d;
    bool ok = read_data(d, path, kind, target_col, err);
    A.data_ok = ok;
    if (!ok) { A.data_msg = err; return false; }
    free_textures(A.data_tex);
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
        fit_data_to_model();
        TrainConfig keep = A.cfg;
        A.plan = autotune(A.sys, A.data);
        A.cfg = keep;
    } else {
        A.model.reset();
        A.from_file = false;
        A.ft_added = A.ft_dropped = 0;
        reset_generation();
        auto_configure();
    }
    return true;
}

void load_example(int kind) {
    if (kind >= 2) {
        // Pictures and melodies are drawn on first use into the settings folder.
        std::string dir = (upath(config_dir()) / "examples" / (kind == 2 ? "pictures" : "melodies")).u8string();
        std::error_code ec;
        std::string err;
        bool ok = std::filesystem::is_directory(upath(dir), ec) && !std::filesystem::is_empty(upath(dir), ec);
        if (!ok) ok = kind == 2 ? write_demo_images(dir, 1000, &err) : write_demo_sounds(dir, 60, &err);
        if (!ok) { A.data_ok = false; A.data_msg = err; return; }
        A.data_kind = kind;
        load_data_impl(dir, kind, -1);
        return;
    }
    std::string p = find_example(kind ? "spiral.csv" : "tiny_text.txt");
    if (p.empty()) { A.data_ok = false; A.data_msg = tr(S::ExampleMissing); }
    else load_data_impl(p, kind, -1);
}

void reset_model() {
    if (busy()) return;
    A.model.reset();
    A.from_file = false;
    A.ft_added = A.ft_dropped = 0;
    reset_generation();
    if (A.data.kind == DATA_TEXT && A.data.media != MEDIA_AUDIO) {
        // Restore the dataset's own vocabulary (or palette).
        std::string err, path = A.data.path;
        Dataset d;
        if (read_data(d, path, A.data.media == MEDIA_IMAGE ? 2 : 0, -1, err)) {
            free_textures(A.data_tex);
            A.data = std::move(d);
        }
    }
    auto_configure();
}

using EnvVars = std::vector<std::pair<std::string, std::string>>;
EnvVars plugin_env();
void scan_all_plugins();
const char* lang_code() {
    static const char* const codes[LANG_COUNT] = {"en", "tr", "ru"};
    return codes[get_language()];
}

// One hook per turned-on skill; logs why skills were skipped.
void add_skill_hooks() {
    A.cfg.hooks.clear();
    A.active_skills.clear();
    if (A.skills_on.empty()) return;
    if (!A.plugins_scanned) scan_all_plugins();
    std::vector<const Plugin*> on;
    for (auto& p : A.plugins)
        if (p.skill && A.skills_on.count(p.file)) on.push_back(&p);
    if (on.empty()) return;
    if (!A.python.searching() && !A.python.found()) A.python.start();
    for (int i = 0; i < 1000 && A.python.searching(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::vector<std::string> python = A.python.command();
    if (python.empty()) {
        log_line(tr(S::SkillsNoPython), State::T_BAD);
        return;
    }
    EnvVars env = plugin_env();
    env.push_back({"KOR_DATA", A.data.path});
    for (const Plugin* p : on) {
        SkillSpec sp;
        sp.name = p->title(get_language());
        sp.script = p->path;
        size_t i = (size_t)(p - A.plugins.data());
        sp.args = split_args(i < A.plugin_args.size() ? A.plugin_args[i].data() : p->args.c_str());
        sp.python = python;
        sp.env = env;
        sp.lang = lang_code();
        sp.q_tag = A.q_tag;
        sp.a_tag = A.a_tag;
        A.cfg.hooks.push_back(std::make_shared<PythonSkill>(sp));
        A.active_skills.push_back(sp.name);
    }
}

void begin_training() {
    if (busy() || A.data.kind == DATA_NONE) return;
    if (!A.model || !model_matches_data()) build_model();
    A.gen_left = 0;
    A.cfg.seed = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
    A.cfg.probes = A.data.kind == DATA_TEXT && A.data.media == MEDIA_TEXT ? current_tests() : std::vector<Probe>();
    A.cfg.probe_target = A.stop_on_pass ? 1.f : 0.f;
    A.run_tests = A.cfg.probes;
    A.results.clear();
    A.train_data = A.data;
    int held = A.train_data.hold_out(A.cfg.probes);
    start_log();
    if (held) log_fmt(State::T_MUTED, S::LogHeldOut, held);
    add_skill_hooks();
    if (!A.active_skills.empty()) {
        std::string names;
        for (auto& n : A.active_skills) names += (names.empty() ? "" : ", ") + n;
        log_line(std::string(tr(S::SkillsActive)) + ": " + names, State::T_ACCENT);
    }
    if (A.train_data.kind == DATA_TEXT && A.train_data.n_train < (size_t)A.model->spec.ctx + 2) {
        log_fmt(State::T_BAD, S::LogError, tr(S::HeldOutTooMuch));
        A.log_file.close();
        return;
    }
    if (!A.trainer.start(A.model, &A.train_data, A.cfg)) {
        log_fmt(State::T_BAD, S::LogError, A.trainer.stats().error.c_str());
        A.log_file.close();
    } else {
        A.settings_open = false;  // the chart, log and tests get the room
    }
}

std::string settings_file() { return (upath(config_dir()) / "settings.cfg").u8string(); }

const ImVec4& state_color(int state) {
    switch (state) {
        case TS_RUNNING: return C::accent;
        case TS_PAUSED: return C::warn;
        case TS_DONE: return C::good;
        case TS_ERROR: return C::bad;
    }
    return C::muted;
}

const char* state_name(int state) {
    static const S names[] = {S::StIdle, S::StRunning, S::StPaused, S::StDone, S::StStopped, S::StError};
    return tr(names[state < 0 || state > 5 ? 0 : state]);
}

// =================================================================== pages

// ------------------------------------------------------------------- home

void step_cards() {
    static const S titles[] = {S::NavData, S::NavModel, S::NavTrain, S::NavTest};
    static const S texts[] = {S::Step1, S::Step2, S::Step3, S::Step4};
    static const int pages[] = {PAGE_DATA, PAGE_MODEL, PAGE_TRAIN, PAGE_TEST};
    static const Icon icons[] = {IC_DATA, IC_MODEL, IC_TRAIN, IC_CHAT};
    float g = px(12), pad = px(16);
    int cols = fit_columns(px(200), 4, g);
    float cw = column_width(cols, g), tw = cw - pad * 2;
    ImFont* bf = F.bold ? F.bold : ImGui::GetFont();
    ImFont* body = ImGui::GetFont();
    float h = 0;
    for (int i = 0; i < 4; ++i)
        h = std::max(h, pad + px(30) + px(14) + bf->FontSize + px(6) + text_size(body, tr(texts[i]), nullptr, tw).y + pad);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(g, g));
    for (int i = 0; i < 4; ++i) {
        if (i % cols) ImGui::SameLine(0, g);
        ImGui::PushID(i);
        ImVec2 p = ImGui::GetCursorScreenPos();
        if (ImGui::InvisibleButton("##step", ImVec2(cw, h))) A.page = pages[i];
        bool hov = ImGui::IsItemHovered();
        if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        float t = anim(ImGui::GetItemID(), hov);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(p, ImVec2(p.x + cw, p.y + h), col(mix(C::surface, C::surface2, t)), px(12));
        dl->AddRect(p, ImVec2(p.x + cw, p.y + h), col(mix(C::border, C::border_hi, t)), px(12));
        ImVec2 nc(p.x + pad + px(15), p.y + pad + px(15));
        dl->AddCircleFilled(nc, px(15), col(C::accent, 0.14f));
        std::string num = std::to_string(i + 1);
        ImFont* nf = F.small_bold ? F.small_bold : body;
        ImVec2 ns = text_size(nf, num.c_str());
        dl->AddText(nf, nf->FontSize, ImVec2(nc.x - ns.x * 0.5f, nc.y - ns.y * 0.5f), col(C::accent), num.c_str());
        icon(dl, icons[i], ImVec2(p.x + cw - pad - px(9), nc.y), px(18), col(mix(C::subtle, C::accent, t)));
        float y = p.y + pad + px(30) + px(14);
        dl->AddText(bf, bf->FontSize, ImVec2(p.x + pad, y), col(C::text), tr(titles[i]));
        dl->AddText(body, body->FontSize, ImVec2(p.x + pad, y + bf->FontSize + px(6)), col(C::muted), tr(texts[i]), nullptr, tw);
        ImGui::PopID();
    }
    ImGui::PopStyleVar();
}

void page_home() {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float w = ImGui::GetContentRegionAvail().x;
    float lz = px(68);
    draw_logo(dl, p, lz);
    ImGui::SetCursorScreenPos(ImVec2(p.x + lz + px(20), p.y + px(4)));
    ImGui::BeginGroup();
    {
        FontScope f(F.h1);
        ImGui::TextUnformatted("Kor Studio");
    }
    {
        FontScope f(F.bold);
        text(tr(S::Tagline), C::accent_hi);
    }
    ImGui::EndGroup();
    ImGui::SetCursorScreenPos(ImVec2(p.x, std::max(p.y + lz, ImGui::GetItemRectMax().y) + px(16)));
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + std::min(w, px(800)));
    text(tr(S::HomeIntro), C::muted);
    ImGui::PopTextWrapPos();
    gap(6);
    if (button(tr(S::Start), BTN_PRIMARY, IC_PLAY, 0, px(40))) A.page = PAGE_DATA;
    ImGui::SameLine(0, px(10));
    ImGui::BeginDisabled(busy());
    if (button(tr(S::ExampleText), BTN_SECONDARY, IC_DOC, 0, px(40))) {
        load_example(0);
        A.page = PAGE_DATA;
    }
    ImGui::EndDisabled();

    gap(22);
    section_title(tr(S::YourSystem));
    text_wrapped(tr(S::GpuNote), C::muted, F.small);
    gap(2);
    std::vector<Stat> sys;
    sys.push_back({tr(S::Cpu), A.sys.cpu, "", C::text, IC_CPU});
    sys.push_back({tr(S::Threads), std::to_string(A.sys.logical), strf(tr(S::ThreadsFmt), A.sys.logical, recommended_threads(A.sys)), C::text, IC_THREADS});
    sys.push_back({tr(S::Ram), strf("%.1f GB", A.sys.ram_total_gb), strf(tr(S::RamFmt), A.sys.ram_total_gb, A.sys.ram_free_gb), C::text, IC_MEMORY});
    sys.push_back({tr(S::Simd), K().name, "", C::good, IC_BOLT});
    if (!A.sys.gpus.empty())
        sys.push_back({tr(S::Gpu), A.sys.gpus[0], A.sys.gpus.size() > 1 ? A.sys.gpus[1] : "", C::text, IC_GPU});
    stat_grid(sys, px(180), false);

    gap(22);
    section_title(tr(S::Workflow));
    gap(2);
    step_cards();

    gap(22);
    begin_card("light", tr(S::LightTitle));
    static const S feats[] = {S::Light1, S::Light2, S::Light3, S::Light4, S::Light5};
    static const Icon ficons[] = {IC_SPARK, IC_LAYERS, IC_TARGET, IC_SAVE, IC_BOLT};
    int cols = ImGui::GetContentRegionAvail().x > px(640) ? 2 : 1;
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(px(10), px(8)));
    if (ImGui::BeginTable("##feats", cols, ImGuiTableFlags_SizingStretchSame)) {
        for (int i = 0; i < 5; ++i) {
            ImGui::TableNextColumn();
            ImVec2 q = ImGui::GetCursorScreenPos();
            float s = px(30);
            ImDrawList* d = ImGui::GetWindowDrawList();
            d->AddRectFilled(q, ImVec2(q.x + s, q.y + s), col(C::accent, 0.12f), px(8));
            icon(d, ficons[i], ImVec2(q.x + s * 0.5f, q.y + s * 0.5f), px(16), col(C::accent));
            ImGui::Dummy(ImVec2(s, s));
            ImGui::SameLine(0, px(12));
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (s - ImGui::GetFontSize()) * 0.5f);
            ImGui::PushTextWrapPos(0.f);
            ImGui::TextUnformatted(tr(feats[i]));
            ImGui::PopTextWrapPos();
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleVar();
    end_card();
}

// ------------------------------------------------------------------- data

void page_data() {
    header(tr(S::NavData), tr(S::DataHelp));
    header_end();

    begin_card("source", tr(S::Source));
    ImGui::BeginDisabled(busy());
    const char* kinds[] = {tr(S::KindText), tr(S::KindTable), tr(S::KindImages), tr(S::KindSound)};
    segmented("kind", &A.data_kind, kinds, 4);
    float sp = ImGui::GetStyle().ItemSpacing.x;
    float bw_browse = has_native_dialogs() ? button_width(tr(S::Browse), IC_FOLDER) + sp : 0;
    float bw_load = button_width(tr(S::Load), IC_UPLOAD);
    ImGui::SetNextItemWidth(std::max(px(120), ImGui::GetContentRegionAvail().x - bw_browse - bw_load - sp));
    bool enter = ImGui::InputTextWithHint("##path", tr(S::PathHint), A.data_path, sizeof A.data_path, ImGuiInputTextFlags_EnterReturnsTrue);
    if (has_native_dialogs()) {
        ImGui::SameLine();
        if (button(tr(S::Browse), BTN_SECONDARY, IC_FOLDER)) {
            std::string p = open_file_dialog(DLG_DATA);
            if (!p.empty()) {
                std::snprintf(A.data_path, sizeof A.data_path, "%s", p.c_str());
                std::string ext = p.size() > 4 ? p.substr(p.size() - 4) : "";
                if (ext == ".csv" || ext == ".tsv") A.data_kind = 1;
                else if (ext == ".txt") A.data_kind = 0;
                else if (is_image_file(p)) A.data_kind = 2;
                else if (is_wav_file(p)) A.data_kind = 3;
                if (A.data_kind == 2) p = upath(p).parent_path().u8string();
                std::snprintf(A.data_path, sizeof A.data_path, "%s", p.c_str());
                enter = true;
            }
        }
    }
    ImGui::SameLine();
    if (button(tr(S::Load), BTN_PRIMARY, IC_UPLOAD) || enter) load_data_impl(A.data_path, A.data_kind, -1);
    // Options of the chosen kind; they apply on the next load.
    float lw = std::min(px(200), ImGui::GetContentRegionAvail().x * 0.3f);
    float sw = std::min(px(260), ImGui::GetContentRegionAvail().x - lw - px(20));
    if (A.data_kind == 0) {
        checkbox(tr(S::Subword), &A.subword);
        tooltip(tr(S::SubwordHelp));
        if (A.subword) {
            ImGui::SameLine(0, px(28));
            ImGui::AlignTextToFramePadding();
            text(tr(S::SubwordSize), C::muted);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(sw);
            slider_int("##subword", &A.subword_size, 256, 4000);
            A.subword_size = std::max(256, std::min(4000, A.subword_size));
        }
    } else if (A.data_kind == 2) {
        field_label(tr(S::ImgSize), lw);
        ImGui::SetNextItemWidth(sw);
        slider_int("##isize", &A.img_size, 8, 32, "%d px");
        field_label(tr(S::ImgColors), lw);
        ImGui::SetNextItemWidth(sw);
        slider_int("##icol", &A.img_colors, 4, 64);
        checkbox(tr(S::Mirror), &A.img_mirror);
        tooltip(tr(S::MirrorHelp));
        A.img_size = std::max(8, std::min(32, A.img_size));
        A.img_colors = std::max(4, std::min(64, A.img_colors));
    } else if (A.data_kind == 3) {
        static const int rates[] = {4000, 8000, 11025, 16000};
        const char* names[] = {"4000 Hz", "8000 Hz", "11025 Hz", "16000 Hz"};
        int cur = 1;
        for (int i = 0; i < 4; ++i)
            if (rates[i] == A.snd_rate) cur = i;
        field_label(tr(S::SampleRate), lw);
        if (segmented("rate", &cur, names, 4)) A.snd_rate = rates[cur];
    }
    if (A.data_kind >= 2) text_wrapped(tr(S::FolderNote), C::muted, F.small);
    ImGui::AlignTextToFramePadding();
    text(tr(S::OrExample), C::muted);
    ImGui::SameLine();
    if (button(tr(S::ExampleText), BTN_GHOST, IC_DOC)) load_example(0);
    ImGui::SameLine(0, px(4));
    if (button(tr(S::ExampleTable), BTN_GHOST, IC_TABLE)) load_example(1);
    ImGui::SameLine(0, px(4));
    if (button(tr(S::ExampleImages), BTN_GHOST, IC_SPARK)) load_example(2);
    ImGui::SameLine(0, px(4));
    if (button(tr(S::ExampleSounds), BTN_GHOST, IC_SPARK)) load_example(3);
    ImGui::EndDisabled();
    if (A.data.items > 0 && A.data.skipped > 0) banner(strf(tr(S::Skipped), A.data.skipped).c_str(), C::warn, IC_WARN);
    if (busy()) banner(tr(S::Busy), C::warn, IC_INFO);
    if (!A.data_ok && !A.data_msg.empty()) banner(strf(tr(S::Failed), A.data_msg.c_str()).c_str(), C::bad, IC_WARN);
    end_card();
    if (A.data.kind == DATA_NONE) return;

    gap();
    std::vector<Stat> st;
    if (A.data.media == MEDIA_IMAGE) {
        st.push_back({tr(S::ImagesLbl), std::to_string(A.data.items), "", C::text, IC_DOC});
        st.push_back({tr(S::ImgSize), strf("%d x %d", A.data.img_size, A.data.img_size), "", C::text, IC_LAYERS});
        st.push_back({tr(S::ColorsLbl), std::to_string(A.data.vocab.size() - 1), "", C::text, IC_SPARK});
        st.push_back({tr(S::Tokens), fmt_count((double)A.data.ids.size()), "", C::blue, IC_PERCENT});
    } else if (A.data.media == MEDIA_AUDIO) {
        st.push_back({tr(S::SoundFiles), std::to_string(A.data.items), "", C::text, IC_DOC});
        st.push_back({tr(S::Seconds), strf("%.1f", (double)A.data.ids.size() / std::max(1, A.data.rate)), "", C::text, IC_CLOCK});
        st.push_back({tr(S::SampleRate), strf("%d Hz", A.data.rate), "", C::text, IC_LAYERS});
        st.push_back({tr(S::Tokens), fmt_count((double)A.data.ids.size()), "", C::blue, IC_PERCENT});
    } else if (A.data.kind == DATA_TEXT) {
        bool sub = A.data.vocab.size() > 0 && std::any_of(A.data.vocab.begin(), A.data.vocab.end(), [](const std::string& v) { return utf8_chars(v).size() > 1; });
        st.push_back({tr(sub ? S::Tokens : S::Characters), fmt_count((double)A.data.ids.size()), "", C::text, IC_DOC});
        st.push_back({tr(S::VocabSize), std::to_string(A.data.vocab.size()), "", C::text, IC_LAYERS});
        st.push_back({tr(S::TrainVal), fmt_count((double)A.data.train_items()) + " / " + fmt_count((double)A.data.val_items()), "", C::text, IC_PERCENT});
        st.push_back({tr(S::Format), tr(A.q_tag.empty() ? S::FmtPlain : S::FmtDialogue),
                      A.q_tag.empty() ? "" : A.q_tag + "/ " + A.a_tag, A.q_tag.empty() ? C::text : C::blue, IC_CHAT});
    } else {
        st.push_back({tr(S::Rows), std::to_string(A.data.train_idx.size() + A.data.val_idx.size()), "", C::text, IC_TABLE});
        st.push_back({tr(S::Features), std::to_string(A.data.n_feat), "", C::text, IC_LAYERS});
        st.push_back({tr(S::Task), A.data.regression ? tr(S::Regression) : strf(tr(S::Classification), (int)A.data.labels.size()), "", C::blue, IC_TARGET});
        st.push_back({tr(S::TrainVal), strf("%d / %d", (int)A.data.train_idx.size(), (int)A.data.val_idx.size()), "", C::text, IC_PERCENT});
    }
    stat_grid(st, px(170), false);

    if (A.data.kind == DATA_TABLE) {
        gap();
        begin_card("target");
        float lw = std::min(px(220), ImGui::GetContentRegionAvail().x * 0.4f);
        ImGui::BeginDisabled(busy());
        field_label(tr(S::Target), lw);
        ImGui::SetNextItemWidth(std::min(px(320), ImGui::GetContentRegionAvail().x));
        if (begin_combo("##target", A.data.header[A.data.target_col].c_str())) {
            for (int c = 0; c < A.data.n_cols; ++c)
                if (ImGui::Selectable(A.data.header[c].c_str(), c == A.data.target_col) && c != A.data.target_col) {
                    std::string path = A.data.path;
                    load_data_impl(path, 1, c);
                    break;
                }
            ImGui::EndCombo();
        }
        ImGui::EndDisabled();
        if (A.data.bad_cells > 0) banner(strf(tr(S::BadCells), A.data.bad_cells).c_str(), C::warn, IC_WARN);
        end_card();
        if (A.data.kind == DATA_NONE) return;
    }

    gap();
    float rest = ImGui::GetContentRegionAvail().y;
    std::string file_name = std::filesystem::u8path(A.data.path).filename().u8string();
    begin_card("preview", tr(S::Preview), 0, 0, file_name.c_str());
    float inner = std::max(px(220), rest - ImGui::GetCursorPosY() - card_padding_y() - px(2));
    if (A.data.media == MEDIA_IMAGE) {
        if (A.data_tex.empty()) {
            size_t n = (size_t)A.data.img_size * A.data.img_size;
            for (size_t s : A.data.layout.starts) {
                if (A.data_tex.size() >= 60 || s + 1 + n > A.data.ids.size()) break;
                A.data_tex.push_back(make_texture(image_from_tokens(A.data.vocab, &A.data.ids[s + 1], n, A.data.img_size)));
            }
        }
        if (begin_inset("##pimg", inner)) image_grid(A.data_tex, px(72));
        ImGui::EndChild();
    } else if (A.data.media == MEDIA_AUDIO) {
        if (begin_inset("##psnd", inner)) {
            size_t n = std::min<size_t>(A.data.ids.size(), (size_t)A.data.rate * 10);
            std::vector<float> wave(n);
            for (size_t i = 0; i < n; ++i) wave[i] = A.data.ids[i] > 0 ? mulaw_decode(A.data.ids[i] - 1) : 0.f;
            waveform(wave, ImVec2(ImGui::GetContentRegionAvail().x, std::min(px(180), inner - px(20))));
        }
        ImGui::EndChild();
    } else if (A.data.kind == DATA_TEXT) {
        size_t n = std::min<size_t>(A.data.raw.size(), 4000);
        while (n < A.data.raw.size() && ((unsigned char)A.data.raw[n] & 0xC0) == 0x80) ++n;  // do not cut a UTF-8 sequence
        if (begin_inset("##ptext", inner)) {
            FontScope f(F.mono);
            ImGui::PushTextWrapPos(0);
            ImGui::TextUnformatted(A.data.raw.data(), A.data.raw.data() + n);
            ImGui::PopTextWrapPos();
        }
        ImGui::EndChild();
    } else {
        int nc = std::min(A.data.n_cols, 60);
        ImGuiTableFlags tf = ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollX |
                             ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit;
        if (ImGui::BeginTable("##ptable", nc, tf, ImVec2(0, inner))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            for (int c = 0; c < nc; ++c) ImGui::TableSetupColumn(A.data.header[c].c_str());
            table_headers();
            for (auto& r : A.data.preview) {
                ImGui::TableNextRow();
                for (int c = 0; c < nc; ++c) {
                    ImGui::TableSetColumnIndex(c);
                    if (c == A.data.target_col) text(r[c].c_str(), C::accent_hi);
                    else ImGui::TextUnformatted(r[c].c_str());
                }
            }
            ImGui::EndTable();
        }
    }
    end_card();
}

// ------------------------------------------------------------------ model

void draw_diagram(const ModelSpec& s) {
    struct Box { std::string top, sub; ImVec4 col; };
    std::vector<Box> boxes;
    if (s.kind == MODEL_GPT) {
        boxes.push_back({tr(S::Input), std::to_string(s.ctx), C::muted});
        boxes.push_back({"Embedding", strf("%d x %d", s.vocab, s.dim), C::teal});
        for (int i = 0; i < s.nlayer && i < 12; ++i) {
            boxes.push_back({"Attention", strf("%d heads", s.heads), C::accent});
            boxes.push_back({"MLP", std::to_string(s.dim * 4), C::blue});
        }
        boxes.push_back({"LayerNorm", "", C::violet});
        boxes.push_back({tr(S::Output), std::to_string(s.vocab), C::muted});
    } else {
        boxes.push_back({tr(S::Input), std::to_string(s.in_dim), C::muted});
        for (auto& l : s.layers) {
            if (l.type == L_DENSE) boxes.push_back({layer_name(l.type), std::to_string(l.units), C::accent});
            else if (l.type == L_DROPOUT) boxes.push_back({layer_name(l.type), strf("%.0f%%", l.p * 100), C::muted});
            else boxes.push_back({layer_name(l.type), "", l.type == L_LAYERNORM ? C::violet : C::blue});
        }
        boxes.push_back({tr(S::Output), std::to_string(s.out_dim), C::muted});
    }
    ImFont* bf = F.bold ? F.bold : ImGui::GetFont();
    ImFont* sf = F.small ? F.small : ImGui::GetFont();
    float h = px(60), g = px(30), pad = px(16);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(px(2), px(6)));
    ImGui::BeginChild("##diagram", ImVec2(0, h + px(12) + ImGui::GetStyle().ScrollbarSize), ImGuiChildFlags_AlwaysUseWindowPadding,
                      ImGuiWindowFlags_HorizontalScrollbar);
    ImGui::PopStyleVar();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p = ImGui::GetCursorScreenPos();
    float x = p.x;
    for (size_t i = 0; i < boxes.size(); ++i) {
        const Box& b = boxes[i];
        ImVec2 t1 = text_size(bf, b.top.c_str()), t2 = text_size(sf, b.sub.c_str());
        float w = std::max(px(76), std::max(t1.x, t2.x) + pad * 2);
        dl->AddRectFilled(ImVec2(x, p.y), ImVec2(x + w, p.y + h), col(b.col, 0.12f), px(10));
        dl->AddRect(ImVec2(x, p.y), ImVec2(x + w, p.y + h), col(b.col, 0.45f), px(10));
        float ty = b.sub.empty() ? p.y + (h - t1.y) * 0.5f : p.y + h * 0.5f - t1.y;
        dl->AddText(bf, bf->FontSize, ImVec2(x + (w - t1.x) * 0.5f, ty), col(C::text), b.top.c_str());
        if (!b.sub.empty())
            dl->AddText(sf, sf->FontSize, ImVec2(x + (w - t2.x) * 0.5f, p.y + h * 0.5f + px(2)), col(mix(b.col, C::text, 0.35f)), b.sub.c_str());
        x += w;
        if (i + 1 < boxes.size()) {
            float y = p.y + h * 0.5f;
            ImU32 c = col(C::border_hi);
            dl->AddLine(ImVec2(x + px(5), y), ImVec2(x + g - px(6), y), c, px(1.5f));
            icon(dl, IC_CHEVRON_RIGHT, ImVec2(x + g - px(8), y), px(12), c);
            x += g;
        }
    }
    ImGui::Dummy(ImVec2(x - p.x + px(2), h));
    ImGui::EndChild();
}

void arch_controls(ModelSpec& s) {
    float lw = std::min(px(230), ImGui::GetContentRegionAvail().x * 0.45f);
    if (s.kind == MODEL_GPT) {
        int per = std::max(4, s.dim / std::max(1, s.heads));
        int max_ctx = std::max(8, std::min(512, (int)A.data.n_train - 2));
        field_label(tr(S::Heads), lw);
        slider_int("##heads", &s.heads, 1, 12);
        field_label(tr(S::HeadDim), lw);
        slider_int("##hdim", &per, 8, 64);
        field_label(tr(S::Layers), lw);
        slider_int("##layers", &s.nlayer, 1, 12);
        field_label(tr(S::Context), lw);
        bool image = A.data.media == MEDIA_IMAGE;
        ImGui::BeginDisabled(image);
        slider_int("##ctx", &s.ctx, 8, image ? s.ctx : max_ctx);
        ImGui::EndDisabled();
        s.heads = std::max(1, std::min(12, s.heads));
        s.nlayer = std::max(1, std::min(12, s.nlayer));
        s.dim = s.heads * std::max(4, std::min(64, per));
        if (image) s.ctx = A.data.img_size * A.data.img_size + 1;
        else s.ctx = std::max(8, std::min(max_ctx, s.ctx));
        text_wrapped(strf(tr(S::Width), s.dim).c_str(), C::muted, F.small);
        if (image) text_wrapped(tr(S::MediaCtxNote), C::muted, F.small);
        return;
    }
    float fh = ImGui::GetFrameHeight(), sp = ImGui::GetStyle().ItemSpacing.x;
    int remove = -1, up = -1;
    for (int i = 0; i < (int)s.layers.size(); ++i) {
        LayerSpec& l = s.layers[i];
        ImGui::PushID(i);
        float x0 = ImGui::GetCursorPosX(), aw = ImGui::GetContentRegionAvail().x;
        ImGui::AlignTextToFramePadding();
        text(strf("%2d", i + 1).c_str(), C::subtle);
        ImGui::SameLine(x0 + px(30));
        ImGui::SetNextItemWidth(px(190));
        if (begin_combo("##type", layer_name(l.type))) {
            for (int t = 0; t < L_TYPE_COUNT; ++t)
                if (ImGui::Selectable(layer_name(t), t == l.type)) l.type = t;
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        if (l.type == L_DENSE) {
            ImGui::SetNextItemWidth(px(110));
            ImGui::InputInt("##units", &l.units, 0, 0);
            l.units = std::max(1, std::min(4096, l.units));
            ImGui::SameLine();
            ImGui::AlignTextToFramePadding();
            text(tr(S::Units), C::muted);
        } else if (l.type == L_DROPOUT) {
            ImGui::SetNextItemWidth(px(220));
            slider_float("##p", &l.p, 0.f, 0.8f, "p = %.2f");
        } else {
            ImGui::Dummy(ImVec2(px(1), fh));
        }
        ImGui::SameLine(x0 + aw - fh * 2 - sp);
        ImGui::BeginDisabled(i == 0);
        if (button("##up", BTN_GHOST, IC_ARROW_UP, fh, fh)) up = i;
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (button("##rm", BTN_GHOST, IC_TRASH, fh, fh)) remove = i;
        ImGui::PopID();
    }
    if (up > 0) std::swap(s.layers[up], s.layers[up - 1]);
    if (remove >= 0) s.layers.erase(s.layers.begin() + remove);
    ImGui::SetNextItemWidth(px(190));
    if (begin_combo("##add", layer_name(A.add_type))) {
        for (int t = 0; t < L_TYPE_COUNT; ++t)
            if (ImGui::Selectable(layer_name(t), t == A.add_type)) A.add_type = t;
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    if (button(tr(S::AddLayer), BTN_SECONDARY, IC_PLUS) && s.layers.size() < 64) {
        LayerSpec l;
        l.type = A.add_type;
        s.layers.push_back(l);
    }
}

void fit_card(float min_h, float width, float* natural) {
    begin_card("fit", tr(S::MemoryFit), min_h, width);
    double mb = (double)estimate_train_bytes(A.spec, A.cfg.batch) / (1024.0 * 1024.0);
    double budget = A.has_plan ? A.plan.budget_mb : 512;
    text(tr(S::Params), C::muted);
    {
        FontScope f(F.stat);
        ImGui::TextUnformatted(fmt_count((double)estimate_params(A.spec)).c_str());
    }
    gap(4);
    text(tr(S::EstRam), C::muted);
    {
        FontScope f(F.bold);
        ImGui::TextUnformatted(fmt_mb(mb).c_str());
    }
    ImGui::SameLine();
    {
        FontScope f(F.small);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + px(2));
        text(strf(tr(S::Budget), budget).c_str(), C::muted);
    }
    float ratio = (float)(mb / std::max(1.0, budget));
    const ImVec4& c = ratio <= 0.7f ? C::good : ratio <= 1.f ? C::warn : C::bad;
    progress_bar(ratio, px(6), c);
    gap(2);
    ImVec2 p = ImGui::GetCursorScreenPos();
    icon(ImGui::GetWindowDrawList(), mb <= budget ? IC_CHECK : IC_WARN, ImVec2(p.x + px(8), p.y + ImGui::GetFontSize() * 0.5f), px(14), col(mb <= budget ? C::good : C::bad));
    ImGui::SetCursorScreenPos(ImVec2(p.x + px(22), p.y));
    ImGui::PushTextWrapPos(0);
    text(tr(mb <= budget ? S::Fits : S::TooBig), mb <= budget ? C::good : C::bad);
    ImGui::PopTextWrapPos();
    end_card(natural);
}

void page_model() {
    char b[256];
    if (A.model) {
        header(tr(S::NavModel), tr(S::ModelHelp), button_width(tr(S::ResetModel), IC_TRASH));
        ImGui::BeginDisabled(busy());
        if (button(tr(S::ResetModel), BTN_DANGER, IC_TRASH)) reset_model();
        ImGui::EndDisabled();
        header_end();
        if (!A.model) return;
        size_t total, train;
        {
            std::lock_guard<std::mutex> lk(A.trainer.model_mu);
            total = A.model->param_count(false);
            train = A.model->param_count(true);
        }
        std::vector<Stat> st;
        const ModelSpec& ms = A.model->spec;
        std::string sub = ms.kind != MODEL_GPT ? ""
                        : ms.media == MEDIA_IMAGE ? strf("%dx%d, %d %s", ms.img_size, ms.img_size, ms.vocab - 1, tr(S::ColorsLbl))
                        : ms.media == MEDIA_AUDIO ? strf("%d Hz", ms.rate) : "";
        S type = ms.kind != MODEL_GPT ? S::TableModel : ms.media == MEDIA_IMAGE ? S::ImageModel : ms.media == MEDIA_AUDIO ? S::SoundModel : S::TextModel;
        st.push_back({tr(S::ModelType), tr(type), sub, C::text, IC_MODEL});
        st.push_back({tr(S::Params), fmt_count((double)total), "", C::text, IC_LAYERS});
        st.push_back({tr(S::Trainable), fmt_count((double)train), "", A.model->spec.lora_rank ? C::accent_hi : C::text, IC_TARGET});
        stat_grid(st, px(200), false);
        gap();
        if (A.from_file && model_matches_data()) {
            std::string ft = tr(S::FineTuneHelp);
            if (A.ft_added) ft += "  " + strf(tr(S::FineTuneAdded), A.ft_added);
            if (A.ft_dropped) ft += "  " + strf(tr(S::FineTuneDropped), A.ft_dropped);
            banner(ft.c_str(), C::blue, IC_SPARK);
            gap();
        }
        begin_card("built", tr(S::Network), 0, 0, tr(S::ResetHint));
        draw_diagram(A.model->spec);
        end_card();
        gap();
        begin_card("lora", tr(S::LoraTitle), 0, 0, tr(S::LoraHelp));
        ImGui::BeginDisabled(busy());
        if (A.model->spec.lora_rank == 0) {
            float bw = button_width(tr(S::EnableLora), IC_PLUS);
            field_label(tr(S::Rank), std::min(px(140), ImGui::GetContentRegionAvail().x * 0.3f));
            ImGui::SetNextItemWidth(std::max(px(80), std::min(px(260), ImGui::GetContentRegionAvail().x - bw - px(16))));
            slider_int("##rank", &A.lora_rank, 1, 32);
            ImGui::SameLine(0, px(16));
            if (button(tr(S::EnableLora), BTN_PRIMARY, IC_PLUS)) A.model->enable_lora(A.lora_rank);
        } else {
            std::snprintf(b, sizeof b, "%s: %d", tr(S::Rank), A.model->spec.lora_rank);
            badge(b, C::accent_hi, true);
            if (button(tr(S::MergeLora), BTN_SECONDARY, IC_LAYERS)) A.model->merge_lora();
        }
        ImGui::EndDisabled();
        end_card();
        return;
    }

    if (A.data.kind == DATA_NONE) {
        header(tr(S::NavModel), tr(S::ModelHelp));
        header_end();
        if (empty_state(IC_DATA, nullptr, tr(S::NoData), tr(S::GoData)) == 1) A.page = PAGE_DATA;
        return;
    }
    finalize_spec();
    float sp = ImGui::GetStyle().ItemSpacing.x;
    header(tr(S::NavModel), tr(S::ModelHelp), button_width(tr(S::AutoBtn), IC_SPARK) + sp + button_width(tr(S::BuildModel), IC_CHECK));
    if (button(tr(S::AutoBtn), BTN_SECONDARY, IC_SPARK)) auto_configure();
    ImGui::SameLine();
    if (button(tr(S::BuildModel), BTN_PRIMARY, IC_CHECK)) build_model();
    header_end();
    if (A.has_plan) {
        badge(strf(tr(S::AutoDone), tier_name(A.plan.tier)).c_str(), C::blue);
        gap(2);
    }

    static EqualRow row;
    float g = px(16);
    bool two = ImGui::GetContentRegionAvail().x > px(780);
    float total = ImGui::GetContentRegionAvail().x;
    float lw = two ? std::floor((total - g) * 0.62f) : 0;
    begin_card("arch", tr(S::Arch), two ? row.prev : 0, lw);
    arch_controls(A.spec);
    end_card(row.track());
    if (two) ImGui::SameLine(0, g);
    else gap();
    fit_card(two ? row.prev : 0, two ? total - lw - g : 0, row.track());
    row.next();
    gap();
    begin_card("diagram", tr(S::Network));
    draw_diagram(A.spec);
    end_card();
}

// --------------------------------------------------------------- training

void plot_loss(const TrainStats& st, ImVec2 size) {
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##plot", size);
    bool hov = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p1(p0.x + size.x, p0.y + size.y);
    dl->AddRectFilled(p0, p1, col(C::surface), px(12));
    dl->AddRect(p0, p1, col(C::border), px(12));
    ImFont* sf = F.small ? F.small : ImGui::GetFont();

    // legend
    float lx = p0.x + px(18), ly = p0.y + px(16);
    for (int k = 0; k < 2; ++k) {
        const char* name = tr(k ? S::ValLoss : S::Loss);
        const ImVec4& c = k ? C::blue : C::accent;
        dl->AddRectFilled(ImVec2(lx, ly + sf->FontSize * 0.5f - px(1.5f)), ImVec2(lx + px(16), ly + sf->FontSize * 0.5f + px(1.5f)), col(c), px(2));
        dl->AddText(sf, sf->FontSize, ImVec2(lx + px(22), ly), col(C::muted), name);
        lx += px(22) + text_size(sf, name).x + px(22);
    }

    if (st.train_hist.size() < 2) {
        const char* msg = tr(S::PlotEmpty);
        ImVec2 ts = ImGui::CalcTextSize(msg);
        ImVec2 c(p0.x + size.x * 0.5f, p0.y + size.y * 0.5f);
        icon(dl, IC_TREND, ImVec2(c.x, c.y - px(22)), px(30), col(C::subtle));
        dl->AddText(ImVec2(c.x - ts.x * 0.5f, c.y + px(4)), col(C::muted), msg);
        return;
    }
    float lo = 1e30f, hi = -1e30f;
    for (float v : st.train_hist) { lo = std::min(lo, v); hi = std::max(hi, v); }
    for (float v : st.val_hist) { lo = std::min(lo, v); hi = std::max(hi, v); }
    if (hi - lo < 1e-6f) hi = lo + 1e-6f;
    float headroom = (hi - lo) * 0.06f;
    lo = std::max(0.f, lo - headroom);
    hi += headroom;
    ImVec2 a(p0.x + px(60), p0.y + px(46)), bq(p1.x - px(22), p1.y - px(36));
    int total = std::max(st.total, (int)st.train_hist.size());
    auto X = [&](float step) { return a.x + (bq.x - a.x) * (step - 1) / std::max(1, total - 1); };
    auto Y = [&](float v) { return bq.y - (bq.y - a.y) * (v - lo) / (hi - lo); };
    char b[64];
    for (int gi = 0; gi <= 4; ++gi) {
        float v = lo + (hi - lo) * gi / 4.f, y = Y(v);
        dl->AddLine(ImVec2(a.x, y), ImVec2(bq.x, y), col(C::border, gi == 0 ? 1.f : 0.55f));
        std::snprintf(b, sizeof b, "%.3f", v);
        ImVec2 ts = text_size(sf, b);
        dl->AddText(sf, sf->FontSize, ImVec2(a.x - ts.x - px(10), y - ts.y * 0.5f), col(C::subtle), b);
    }
    for (int gi = 0; gi <= 4; ++gi) {
        int step = gi == 0 ? 1 : (int)std::lround((double)total * gi / 4);
        float x = X((float)step);
        std::snprintf(b, sizeof b, "%d", step);
        ImVec2 ts = text_size(sf, b);
        float tx = std::min(std::max(x - ts.x * 0.5f, a.x - px(4)), bq.x - ts.x + px(4));
        dl->AddText(sf, sf->FontSize, ImVec2(tx, bq.y + px(10)), col(C::subtle), b);
    }

    int n = (int)st.train_hist.size();
    int stride = std::max(1, n / std::max(1, (int)(bq.x - a.x)));
    std::vector<ImVec2> pts;
    for (int i = 0; i < n; i += stride) pts.push_back(ImVec2(X((float)(i + 1)), Y(st.train_hist[i])));
    pts.push_back(ImVec2(X((float)n), Y(st.train_hist[n - 1])));

    // area under the training curve, fading towards the axis
    ImVec2 uv = ImGui::GetFontTexUvWhitePixel();
    float span = std::max(1.f, bq.y - a.y);
    for (size_t i = 0; i + 1 < pts.size(); ++i) {
        ImU32 c0 = col(C::accent, 0.20f * (bq.y - pts[i].y) / span), c1 = col(C::accent, 0.20f * (bq.y - pts[i + 1].y) / span);
        ImU32 cb = col(C::accent, 0.f);
        dl->PrimReserve(6, 4);
        ImDrawIdx idx = (ImDrawIdx)dl->_VtxCurrentIdx;
        dl->PrimWriteVtx(pts[i], uv, c0);
        dl->PrimWriteVtx(pts[i + 1], uv, c1);
        dl->PrimWriteVtx(ImVec2(pts[i + 1].x, bq.y), uv, cb);
        dl->PrimWriteVtx(ImVec2(pts[i].x, bq.y), uv, cb);
        dl->PrimWriteIdx(idx); dl->PrimWriteIdx((ImDrawIdx)(idx + 1)); dl->PrimWriteIdx((ImDrawIdx)(idx + 2));
        dl->PrimWriteIdx(idx); dl->PrimWriteIdx((ImDrawIdx)(idx + 2)); dl->PrimWriteIdx((ImDrawIdx)(idx + 3));
    }
    dl->AddPolyline(pts.data(), (int)pts.size(), col(C::accent), 0, px(2.f));
    if (!st.val_hist.empty()) {
        std::vector<ImVec2> vp;
        for (size_t i = 0; i < st.val_hist.size(); ++i) vp.push_back(ImVec2(X((float)st.val_steps[i]), Y(st.val_hist[i])));
        if (vp.size() >= 2) dl->AddPolyline(vp.data(), (int)vp.size(), col(C::blue), 0, px(2.f));
        for (const ImVec2& q : vp) {
            dl->AddCircleFilled(q, px(3.5f), col(C::surface));
            dl->AddCircle(q, px(3.5f), col(C::blue), 0, px(1.6f));
        }
    }

    // hover read-out
    ImVec2 m = ImGui::GetIO().MousePos;
    if (hov && m.x >= a.x && m.x <= bq.x) {
        int step = (int)std::lround((m.x - a.x) / std::max(1.f, bq.x - a.x) * (total - 1) + 1);
        int idx = std::max(0, std::min(n - 1, step - 1));
        float x = X((float)(idx + 1));
        dl->AddLine(ImVec2(x, a.y), ImVec2(x, bq.y), col(C::border_hi), 1.f);
        ImVec2 tp(x, Y(st.train_hist[idx]));
        dl->AddCircleFilled(tp, px(5), col(C::surface));
        dl->AddCircle(tp, px(5), col(C::accent), 0, px(2));
        std::string l1 = strf(tr(S::StepLbl), idx + 1);
        std::string l2 = strf("%s  %.4f", tr(S::Loss), st.train_hist[idx]);
        std::string l3;
        if (!st.val_hist.empty()) {
            size_t best = 0;
            for (size_t i = 1; i < st.val_steps.size(); ++i)
                if (std::abs(st.val_steps[i] - (idx + 1)) < std::abs(st.val_steps[best] - (idx + 1))) best = i;
            ImVec2 vq(X((float)st.val_steps[best]), Y(st.val_hist[best]));
            dl->AddCircleFilled(vq, px(5), col(C::surface));
            dl->AddCircle(vq, px(5), col(C::blue), 0, px(2));
            l3 = strf("%s  %.4f", tr(S::ValLoss), st.val_hist[best]);
        }
        ImFont* bf = F.small_bold ? F.small_bold : sf;
        float tw = std::max(text_size(bf, l1.c_str()).x, std::max(text_size(sf, l2.c_str()).x, text_size(sf, l3.c_str()).x)) + px(34);
        float th = px(12) * 2 + bf->FontSize + (sf->FontSize + px(4)) * (l3.empty() ? 1 : 2);
        float bx = x + px(14);
        if (bx + tw > p1.x - px(8)) bx = x - px(14) - tw;
        float by = std::max(a.y, std::min(m.y - th * 0.5f, bq.y - th));
        dl->AddRectFilled(ImVec2(bx, by), ImVec2(bx + tw, by + th), col(C::surface3), px(8));
        dl->AddRect(ImVec2(bx, by), ImVec2(bx + tw, by + th), col(C::border_hi), px(8));
        float ty = by + px(12);
        dl->AddText(bf, bf->FontSize, ImVec2(bx + px(12), ty), col(C::text), l1.c_str());
        ty += bf->FontSize + px(4);
        dl->AddCircleFilled(ImVec2(bx + px(16), ty + sf->FontSize * 0.5f), px(3.5f), col(C::accent));
        dl->AddText(sf, sf->FontSize, ImVec2(bx + px(26), ty), col(C::text), l2.c_str());
        if (!l3.empty()) {
            ty += sf->FontSize + px(4);
            dl->AddCircleFilled(ImVec2(bx + px(16), ty + sf->FontSize * 0.5f), px(3.5f), col(C::blue));
            dl->AddText(sf, sf->FontSize, ImVec2(bx + px(26), ty), col(C::text), l3.c_str());
        }
    }
}

const ImVec4& tone_color(int tone) {
    switch (tone) {
        case State::T_GOOD: return C::good;
        case State::T_BAD: return C::bad;
        case State::T_ACCENT: return C::accent_hi;
        case State::T_MUTED: return C::muted;
    }
    return C::text;
}

void log_view(float height) {
    ImGui::BeginDisabled(A.log.empty());
    if (button(tr(S::CopyLog), BTN_GHOST, IC_COPY)) {
        std::string all;
        for (auto& l : A.log) all += l.text + "\n";
        ImGui::SetClipboardText(all.c_str());
    }
    ImGui::EndDisabled();
    ImGui::SameLine(0, px(4));
    if (button(tr(S::OpenLogs), BTN_GHOST, IC_FOLDER)) {
        std::error_code ec;
        std::filesystem::create_directories(upath(logs_dir()), ec);
        open_folder(logs_dir());
    }
    if (!A.log_path.empty()) {
        ImGui::SameLine(0, px(12));
        ImVec2 p = ImGui::GetCursorScreenPos();
        float w = ImGui::GetContentRegionAvail().x, fh = ImGui::GetFrameHeight();
        ImFont* sf = F.small ? F.small : ImGui::GetFont();
        std::string s = strf(tr(S::LogFile), std::filesystem::u8path(A.log_path).filename().u8string().c_str());
        ImGui::Dummy(ImVec2(std::max(1.f, w), fh));
        text_ellipsis(ImGui::GetWindowDrawList(), sf, ImVec2(p.x, p.y + (fh - sf->FontSize) * 0.5f), w, s.c_str(), col(C::subtle));
        tooltip(A.log_path.c_str());
    }
    float h = std::max(px(140), height - ImGui::GetFrameHeight() - ImGui::GetStyle().ItemSpacing.y);
    if (begin_inset("##log", h, true)) {
        FontScope f(F.mono);
        if (A.log.empty()) text(tr(S::LogEmpty), C::subtle);
        bool at_end = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4;
        ImGuiListClipper clip;
        clip.Begin((int)A.log.size());
        while (clip.Step())
            for (int i = clip.DisplayStart; i < clip.DisplayEnd; ++i) {
                const std::string& t = A.log[i].text;
                size_t close = t.size() > 1 && t[0] == '[' ? t.find("] ") : std::string::npos;
                if (close != std::string::npos && close < 12) {
                    text(t.substr(0, close + 1).c_str(), C::subtle);
                    ImGui::SameLine(0, ImGui::CalcTextSize(" ").x);
                    text(t.c_str() + close + 2, tone_color(A.log[i].tone));
                } else {
                    text(t.c_str(), tone_color(A.log[i].tone));
                }
            }
        if (at_end) ImGui::SetScrollHereY(1.f);
    }
    ImGui::EndChild();
}

void history_bars(const TrainStats& st, float height) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    float w = ImGui::GetContentRegionAvail().x;
    ImGui::Dummy(ImVec2(w, height));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    int n = (int)st.probe_hist.size();
    float total = (float)std::max(1, st.probe_total);
    for (int g = 0; g <= 2; ++g) {
        float y = p.y + height - height * g / 2.f;
        dl->AddLine(ImVec2(p.x, y), ImVec2(p.x + w, y), col(C::border, g == 0 ? 1.f : 0.5f));
    }
    if (n == 0) return;
    float slot = w / n, bw = std::max(px(2), std::min(px(22), slot * 0.62f));
    for (int i = 0; i < n; ++i) {
        float f = st.probe_hist[i] / total;
        float x = p.x + slot * i + (slot - bw) * 0.5f;
        float top = p.y + height - std::max(px(5), (height - px(2)) * f);
        const ImVec4& c = f >= 1.f ? C::good : f > 0 ? C::accent : C::bad;
        dl->AddRectFilled(ImVec2(x, top), ImVec2(x + bw, p.y + height), col(c, i == n - 1 ? 1.f : 0.65f), px(3), ImDrawFlags_RoundCornersTop);
    }
}

void tests_view(const TrainStats& st) {
    bool run = busy();
    std::vector<Probe> tests = current_tests();
    int pass = 0;
    for (auto& r : A.results) pass += r.pass;
    int nres = (int)A.results.size();

    // ---- summary
    if (nres > 0) {
        float frac = (float)pass / nres;
        const ImVec4& c = pass == nres ? C::good : pass > 0 ? C::accent : C::bad;
        ring(frac, px(34), px(7), c, strf("%d/%d", pass, nres).c_str());
        ImGui::SameLine(0, px(18));
        ImGui::BeginGroup();
        ImGui::Dummy(ImVec2(0, px(8)));
        {
            FontScope f(F.h2);
            text(strf(tr(S::TestsPassed), pass, nres).c_str(), c);
        }
        text(A.results_step > 0 ? strf(tr(S::ResultsAt), A.results_step).c_str() : tr(S::ResultsManual), C::muted);
        ImGui::EndGroup();
    } else {
        banner(tr(S::NoResults), C::muted, IC_INFO);
    }
    gap(4);

    // ---- editor and history
    static EqualRow row;
    bool hist = st.probe_hist.size() >= 2 && A.results_step > 0;
    float g = px(16), total = ImGui::GetContentRegionAvail().x;
    bool two = hist && total > px(820);
    float lw = two ? std::floor((total - g) * 0.6f) : 0;
    begin_card("editor", tr(S::TabTests), two ? row.prev : 0, lw, tr(S::TestsHelp));
    if (!A.q_tag.empty()) text_wrapped(strf(tr(S::DialogDetected), A.q_tag.c_str(), A.a_tag.c_str()).c_str(), C::blue, F.small);
    else text_wrapped(tr(S::NoDialog), C::muted, F.small);
    ImGui::BeginDisabled(run);
    {
        FontScope f(F.mono);
        ImGui::InputTextMultiline("##tests", A.tests, sizeof A.tests, ImVec2(-FLT_MIN, px(130)));
    }
    ImGui::EndDisabled();
    float sp = ImGui::GetStyle().ItemSpacing.x;
    float bw = button_width(tr(S::RunTestsNow), IC_PLAY) + (A.q_tag.empty() ? 0 : button_width(tr(S::UseDataQuestions), IC_REFRESH) + sp);
    float x0 = ImGui::GetCursorPosX(), aw = ImGui::GetContentRegionAvail().x;
    ImGui::AlignTextToFramePadding();
    text(strf(tr(S::TestsCount), (int)tests.size()).c_str(), C::muted);
    float count_end = ImGui::GetItemRectMax().x - ImGui::GetWindowPos().x + px(12);
    ImGui::SameLine(std::max(count_end, x0 + aw - bw));
    if (!A.q_tag.empty()) {
        ImGui::BeginDisabled(run);
        if (button(tr(S::UseDataQuestions), BTN_SECONDARY, IC_REFRESH)) tests_from_data();
        ImGui::EndDisabled();
        ImGui::SameLine();
    }
    ImGui::BeginDisabled(!can_run_tests() || tests.empty());
    if (button(tr(S::RunTestsNow), BTN_PRIMARY, IC_PLAY)) run_tests_now();
    ImGui::EndDisabled();
    if (tests.empty()) text_wrapped(tr(S::NoTests), C::muted, F.small);
    else if (!A.model || A.model->spec.kind != MODEL_GPT) text_wrapped(tr(S::NeedTextModel), C::muted, F.small);
    end_card(row.track());
    if (hist) {
        if (two) ImGui::SameLine(0, g);
        else gap();
        begin_card("history", tr(S::TestsHistory), two ? row.prev : 0, two ? total - lw - g : 0,
                   strf(tr(S::Latest), st.probe_hist.back(), st.probe_total).c_str());
        float hh = two ? std::max(px(90), row.prev - ImGui::GetCursorPosY() - card_padding_y()) : px(110);
        history_bars(st, hh);
        end_card(nullptr);
        row.cur = std::max(row.cur, px(180));
    }
    row.next();

    // ---- results
    int n = (int)std::min(A.results.size(), A.run_tests.size());
    if (n > 0) {
        gap();
        ImGuiTableFlags tf = ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable |
                             ImGuiTableFlags_SizingStretchProp;
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(px(12), px(9)));
        if (ImGui::BeginTable("##results", 4, tf)) {
            ImGui::TableSetupColumn(tr(S::ColQuestion), ImGuiTableColumnFlags_WidthStretch, 1.2f);
            ImGui::TableSetupColumn(tr(S::ColExpected), ImGuiTableColumnFlags_WidthStretch, 1.f);
            ImGui::TableSetupColumn(tr(S::ColAnswer), ImGuiTableColumnFlags_WidthStretch, 1.f);
            float rw = std::max(badge_width(tr(S::Pass)), badge_width(tr(S::Fail))) + px(8);
            ImGui::TableSetupColumn(tr(S::Result), ImGuiTableColumnFlags_WidthFixed, std::max(rw, ImGui::CalcTextSize(tr(S::Result)).x + px(8)));
            table_headers();
            for (int i = 0; i < n; ++i) {
                const ProbeResult& r = A.results[i];
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::PushTextWrapPos(0);
                ImGui::TextUnformatted(A.run_tests[i].question.c_str());
                ImGui::TableNextColumn();
                text(A.run_tests[i].expected.c_str(), C::muted);
                ImGui::TableNextColumn();
                text(r.answer.empty() ? "-" : r.answer.c_str(), r.pass ? C::good : C::bad);
                ImGui::PopTextWrapPos();
                ImGui::TableNextColumn();
                badge(tr(r.pass ? S::Pass : S::Fail), r.pass ? C::good : C::bad);
            }
            ImGui::EndTable();
        }
        ImGui::PopStyleVar();
    }
}

// The skills of this run (or the turned-on ones before training) and what they changed.
void skills_strip(const TrainStats& st, bool run) {
    std::vector<std::string> names = A.active_skills;
    if (names.empty() && !run && st.state == TS_IDLE)
        for (auto& p : A.plugins)
            if (p.skill && A.skills_on.count(p.file)) names.push_back(p.title(get_language()));
    if (names.empty()) return;
    gap(2);
    ImFont* sb = F.small_bold ? F.small_bold : ImGui::GetFont();
    ImFont* sf = F.small ? F.small : ImGui::GetFont();
    float bh = sb->FontSize + px(8), rowgap = px(6);
    float x0 = ImGui::GetCursorPosX(), right = x0 + ImGui::GetContentRegionAvail().x, y = ImGui::GetCursorPosY();
    ImGui::SetCursorPosY(y + (bh - sb->FontSize) * 0.5f);
    {
        FontScope f(F.small_bold);
        text(tr(S::SkillsActive), C::muted);
    }
    float x = ImGui::GetItemRectMax().x - ImGui::GetWindowPos().x + ImGui::GetScrollX() + px(10);
    auto place = [&](float w) {
        if (x + w > right && x > x0 + px(1)) {
            x = x0;
            y += bh + rowgap;
        }
        ImGui::SetCursorPos(ImVec2(x, y));
        x += w + px(6);
    };
    for (auto& n : names) {
        place(badge_width(n.c_str()));
        badge(n.c_str(), C::violet, true, run);
    }
    std::string info;
    if (st.added_chars > 0) info = strf(tr(S::AddedText), fmt_count((double)st.added_chars).c_str());
    if (st.lr_fixed > 0) info += (info.empty() ? "" : "   \xC2\xB7   ") + strf(tr(S::LrFixed), (double)st.lr_fixed);
    else if (std::fabs(st.lr_factor - 1.f) > 1e-4f) info += (info.empty() ? "" : "   \xC2\xB7   ") + strf(tr(S::LrFactor), (double)st.lr_factor);
    if (!info.empty()) {
        place(text_size(sf, info.c_str()).x + px(4));
        ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPosX() + px(4), y + (bh - sf->FontSize) * 0.5f));
        FontScope f(F.small);
        text(info.c_str(), C::muted);
    }
    ImGui::SetCursorPos(ImVec2(x0, y + bh));
    ImGui::Dummy(ImVec2(0, 0));
}

// Line chart of one skill metric; the value range is labelled in a gutter on the left.
void metric_chart(const TrainStats::Metric& m, ImVec2 size, const ImVec4& c) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Dummy(size);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    size_t n = std::min(m.values.size(), m.steps.size());
    float lo = n ? *std::min_element(m.values.begin(), m.values.begin() + n) : 0.f;
    float hi = n ? *std::max_element(m.values.begin(), m.values.begin() + n) : 1.f;
    if (hi - lo < 1e-6f) { lo -= 0.5f; hi += 0.5f; }
    ImFont* sf = F.small ? F.small : ImGui::GetFont();
    std::string top = strf("%.4g", (double)hi), bottom = strf("%.4g", (double)lo);
    float gutter = std::max(text_size(sf, top.c_str()).x, text_size(sf, bottom.c_str()).x) + px(10);
    float x0 = p.x + gutter, w = std::max(px(10), size.x - gutter), r = px(5);
    float y0 = p.y + r, h = size.y - 2 * r;
    for (int g = 0; g <= 2; ++g) {
        float yy = y0 + h * g / 2.f;
        dl->AddLine(ImVec2(x0, yy), ImVec2(x0 + w, yy), col(C::border, g == 1 ? 0.4f : 0.8f));
    }
    dl->AddText(sf, sf->FontSize, ImVec2(p.x, y0 - sf->FontSize * 0.5f), col(C::subtle), top.c_str());
    dl->AddText(sf, sf->FontSize, ImVec2(p.x, y0 + h - sf->FontSize * 0.5f), col(C::subtle), bottom.c_str());
    if (n == 0) return;
    float s0 = (float)m.steps.front(), s1 = std::max((float)m.steps.back(), s0 + 1.f);
    std::vector<ImVec2> pts(n);
    for (size_t i = 0; i < n; ++i)
        pts[i] = ImVec2(x0 + (m.steps[i] - s0) / (s1 - s0) * (w - r), y0 + h - (m.values[i] - lo) / (hi - lo) * h);
    if (n == 1) pts[0].x = x0 + w * 0.5f;
    if (n > 1) dl->AddPolyline(pts.data(), (int)n, col(c), 0, px(2));
    dl->AddCircleFilled(pts.back(), px(4), col(c));
}

// Values reported by the skills, one small chart each.
void skill_metrics(const TrainStats& st) {
    if (st.metrics.empty()) {
        banner(tr(S::SkillMetricsEmpty), C::muted, IC_INFO);
        return;
    }
    static const ImVec4 colors[] = {C::violet, C::teal, C::blue, C::accent, C::good};
    float g = px(16);
    int cols = fit_columns(px(320), 3, g);
    float cw = column_width(cols, g);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(g, g));
    for (int i = 0; i < (int)st.metrics.size(); ++i) {
        const TrainStats::Metric& m = st.metrics[i];
        if (i % cols) ImGui::SameLine(0, g);
        ImGui::PushID(i);
        std::string sub = m.values.empty() ? "" : strf("%.4g   \xC2\xB7   ", (double)m.values.back()) + strf(tr(S::StepLbl), m.steps.back());
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(px(10), px(10)));
        begin_card("metric", m.name.c_str(), 0, cw, sub.c_str());
        metric_chart(m, ImVec2(ImGui::GetContentRegionAvail().x, px(110)), colors[i % 5]);
        end_card();
        ImGui::PopStyleVar();
        ImGui::PopID();
    }
    ImGui::PopStyleVar();
}

void train_settings(bool run) {
    ImGui::BeginDisabled(run);
    int cols = ImGui::GetContentRegionAvail().x > px(720) ? 2 : 1;
    ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(px(12), 0));
    if (ImGui::BeginTable("##cfg", cols, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextColumn();
        float lw = std::min(px(220), ImGui::GetContentRegionAvail().x * 0.5f);
        field_label(tr(S::Steps), lw);
        ImGui::InputInt("##steps", &A.cfg.steps, 0, 0);
        field_label(tr(S::BatchSize), lw);
        ImGui::InputInt("##batch", &A.cfg.batch, 0, 0);
        field_label(tr(S::Accum), lw);
        ImGui::InputInt("##accum", &A.cfg.accum, 0, 0);
        field_label(tr(S::EvalEvery), lw);
        ImGui::InputInt("##eval", &A.cfg.eval_every, 0, 0);
        field_label(tr(S::Patience), lw, tr(S::PatienceHelp));
        ImGui::InputInt("##patience", &A.cfg.patience, 0, 0);
        if (cols == 2) ImGui::TableNextColumn();
        field_label(tr(S::LR), lw);
        ImGui::InputFloat("##lr", &A.cfg.lr, 0, 0, "%.5f");
        field_label(tr(S::WD), lw);
        ImGui::InputFloat("##wd", &A.cfg.weight_decay, 0, 0, "%.4f");
        field_label(tr(S::ThreadsUse), lw);
        slider_int("##threads", &A.cfg.threads, 1, std::max(1, A.sys.logical));
        const char* opts[] = {"AdamW", "SGD + momentum"};
        field_label(tr(S::Optimizer), lw);
        combo("##opt", &A.cfg.optimizer, opts, 2);
        if (A.data.kind == DATA_TEXT) {
            if (A.data.media == MEDIA_TEXT) {
                field_label(tr(S::TestEvery), lw, tr(S::TestEveryHelp));
                ImGui::InputInt("##probe", &A.cfg.probe_every, 0, 0);
            }
            field_label(tr(S::DropoutLbl), lw, tr(S::DropoutHelp));
            slider_float("##dropout", &A.cfg.dropout, 0.f, 0.5f);
            if (!A.q_tag.empty()) {
                field_label(tr(S::QuestionWeight), lw, tr(S::QuestionWeightHelp));
                slider_float("##qweight", &A.cfg.prompt_weight, 0.f, 1.f);
            }
        }
        ImGui::EndTable();
    }
    ImGui::PopStyleVar();
    gap(2);
    checkbox(tr(S::KeepBest), &A.cfg.keep_best);
    tooltip(tr(S::KeepBestHelp));
    if (A.data.kind == DATA_TEXT && A.data.media == MEDIA_TEXT) {
        ImGui::SameLine(0, px(28));
        checkbox(tr(S::StopOnPass), &A.stop_on_pass);
    }
    bool ema = A.cfg.ema > 0;
    if (checkbox(tr(S::EmaLbl), &ema)) A.cfg.ema = ema ? 0.999f : 0.f;
    tooltip(tr(S::EmaHelp));
    if (A.data.kind == DATA_TEXT && !A.q_tag.empty()) {
        bool align = A.cfg.align > 0;
        ImGui::SameLine(0, px(28));
        if (checkbox(tr(S::AlignRows), &align)) A.cfg.align = align ? 0.75f : 0.f;
        tooltip(tr(S::AlignHelp));
    }
    A.cfg.patience = std::max(0, std::min(1000, A.cfg.patience));
    A.cfg.probe_every = std::max(0, A.cfg.probe_every);
    A.cfg.steps = std::max(1, std::min(10000000, A.cfg.steps));
    A.cfg.batch = std::max(1, std::min(4096, A.cfg.batch));
    A.cfg.accum = std::max(1, std::min(256, A.cfg.accum));
    A.cfg.eval_every = std::max(1, A.cfg.eval_every);
    A.cfg.lr = std::max(1e-7f, std::min(1.f, A.cfg.lr));
    A.cfg.weight_decay = std::max(0.f, std::min(1.f, A.cfg.weight_decay));
    A.cfg.dropout = std::max(0.f, std::min(0.5f, A.cfg.dropout));
    A.cfg.prompt_weight = std::max(0.f, std::min(1.f, A.cfg.prompt_weight));
    A.cfg.threads = std::max(1, std::min(std::max(1, A.sys.logical), A.cfg.threads));
    A.cfg.warmup = std::min(100, A.cfg.steps / 10);
    ImGui::EndDisabled();
    const ModelSpec& ms = A.model && model_matches_data() ? A.model->spec : A.spec;
    double mb = (double)estimate_train_bytes(ms, A.cfg.batch) / (1024.0 * 1024.0);
    std::string info = strf(tr(S::EffBatch), A.cfg.batch * A.cfg.accum) + "   \xC2\xB7   " + tr(S::EstRam) + ": " + fmt_mb(mb);
    text_wrapped(info.c_str(), C::subtle, F.small);
}

void page_train() {
    if (A.data.kind == DATA_NONE) {
        header(tr(S::NavTrain), tr(S::TrainHelp));
        header_end();
        if (empty_state(IC_DATA, nullptr, tr(S::NoData), tr(S::GoData)) == 1) A.page = PAGE_DATA;
        return;
    }
    TrainStats st = A.trainer.stats();
    bool run = busy();
    float sp = ImGui::GetStyle().ItemSpacing.x;
    const char* main_label = tr(A.model && model_matches_data() ? S::ContinueTrain : S::StartTrain);
    bool paused = A.trainer.paused();
    const char* pause_label = tr(paused ? S::Resume : S::Pause);
    float aw = run ? button_width(pause_label, IC_PAUSE) + sp + button_width(tr(S::Stop), IC_STOP) : button_width(main_label, IC_PLAY);
    header(tr(S::NavTrain), tr(S::TrainHelp), aw);
    if (!run) {
        if (button(main_label, BTN_PRIMARY, IC_PLAY)) begin_training();
    } else {
        if (button(pause_label, BTN_SECONDARY, paused ? IC_PLAY : IC_PAUSE)) A.trainer.pause(!paused);
        ImGui::SameLine();
        if (button(tr(S::Stop), BTN_DANGER, IC_STOP)) A.trainer.stop();
    }
    header_end();

    // ---- status
    begin_card("status");
    {
        ImVec2 p = ImGui::GetCursorScreenPos();
        float w = ImGui::GetContentRegionAvail().x;
        const char* sname = state_name(st.state);
        badge(sname, state_color(st.state), true, st.state == TS_RUNNING);
        float bh = ImGui::GetItemRectSize().y;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        std::string steps = strf(tr(S::StepOf), st.step, st.total);
        float fs = ImGui::GetFontSize();
        dl->AddText(ImVec2(p.x + badge_width(sname) + px(12), p.y + (bh - fs) * 0.5f), col(C::muted), steps.c_str());
        std::string pct = strf("%.0f%%", st.total > 0 ? 100.0 * st.step / st.total : 0.0);
        ImFont* bf = F.bold ? F.bold : ImGui::GetFont();
        ImVec2 ps = text_size(bf, pct.c_str());
        dl->AddText(bf, bf->FontSize, ImVec2(p.x + w - ps.x, p.y + (bh - ps.y) * 0.5f), col(C::text), pct.c_str());
    }
    if (st.state == TS_ERROR) text_wrapped(st.error.c_str(), C::bad);
    progress_bar(st.total > 0 ? (float)st.step / st.total : 0.f, px(8), st.state == TS_DONE ? C::good : C::accent);
    skills_strip(st, run);
    end_card();
    gap();

    // ---- metrics
    std::vector<Stat> m;
    m.push_back({tr(S::Loss), st.step ? strf("%.4f", st.loss) : "-", "", C::accent, IC_TREND});
    m.push_back({tr(S::ValLoss), st.val_loss >= 0 ? strf("%.4f", st.val_loss) : "-", "", C::blue, IC_TARGET});
    m.push_back({tr(S::Accuracy), st.val_acc >= 0 ? strf("%.1f%%", st.val_acc * 100) : "-", "", C::text, IC_PERCENT});
    m.push_back({tr(S::Speed), st.step ? fmt_count(st.items_per_sec) : "-", tr(A.data.kind == DATA_TEXT ? S::TokPerSec : S::SampPerSec), C::text, IC_GAUGE});
    m.push_back({tr(S::Elapsed), fmt_time(st.elapsed), "", C::text, IC_CLOCK});
    m.push_back({tr(S::Eta), st.state == TS_RUNNING || st.state == TS_PAUSED ? fmt_time(st.eta) : "-", "", C::text, IC_CLOCK});
    stat_grid(m, px(150), true);
    gap();

    // ---- settings
    std::string summary = strf("%s %d   \xC2\xB7   %s %d   \xC2\xB7   %s %g", tr(S::Steps), A.cfg.steps, tr(S::BatchSize), A.cfg.batch, tr(S::LR), (double)A.cfg.lr);
    if (begin_collapsible_card("cfg", tr(S::Settings), &A.settings_open, summary.c_str())) train_settings(run);
    end_card();
    gap();

    // ---- chart / log / tests / skills
    bool text_data = A.data.kind == DATA_TEXT && A.data.media == MEDIA_TEXT;
    bool skills_tab = !st.metrics.empty() || !A.active_skills.empty();
    std::vector<const char*> labels = {tr(S::TabChart), tr(S::TabLog)};
    std::vector<int> ids = {0, 1};
    if (text_data) { labels.push_back(tr(S::TabTests)); ids.push_back(2); }
    if (skills_tab) { labels.push_back(tr(S::TabSkills)); ids.push_back(3); }
    int sel = 0;
    for (int i = 0; i < (int)ids.size(); ++i)
        if (ids[i] == A.train_tab) sel = i;
    tabs("traintabs", &sel, labels.data(), (int)labels.size());
    A.train_tab = ids[sel];
    float h = std::max(px(300), ImGui::GetContentRegionAvail().y - px(2));
    if (A.train_tab == 0) plot_loss(st, ImVec2(ImGui::GetContentRegionAvail().x, h));
    else if (A.train_tab == 1) log_view(h);
    else if (A.train_tab == 2) tests_view(st);
    else skill_metrics(st);
}

// ------------------------------------------------------------- playground

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

bool load_model_file(const std::string& path) {
    std::string err;
    auto m = Model::load(path, &err);
    A.file_ok = (bool)m;
    A.file_msg = m ? tr(S::LoadedModel) : err;
    if (!m) return false;
    A.model = m;
    A.spec = m->spec;
    reset_generation();
    A.feat_in.clear();
    A.from_file = true;
    if (A.data.kind == DATA_TEXT && A.data.media != MEDIA_TEXT && A.data.media == m->spec.media && !model_matches_data()) {
        // Read the images or sound again in the format of this model.
        std::string e, dp = A.data.path;
        Dataset d;
        if (read_data(d, dp, A.data.media == MEDIA_IMAGE ? 2 : 3, -1, e)) {
            free_textures(A.data_tex);
            A.data = std::move(d);
        }
    }
    fit_data_to_model();
    if (A.data.kind != DATA_NONE && model_matches_data()) A.cfg.lr = A.plan.cfg.lr / 3;
    return true;
}

// Pictures or sound from a snapshot of the model, on a worker thread, so
// neither training nor the interface waits for it.
void start_media_gen() {
    if (A.mg.running || !A.model) return;
    stop_media_gen();
    free_textures(A.gen_tex);
    {
        std::lock_guard<std::mutex> lk(A.mg.mu);
        A.mg.images.clear();
        A.mg.sound = Sound();
    }
    A.media_msg.clear();
    std::string snap = (upath(config_dir()) / "playground-model.tmp").u8string(), err;
    bool saved;
    {
        std::lock_guard<std::mutex> lk(A.trainer.model_mu);
        saved = A.model->save(snap, false, &err);
    }
    if (!saved) { A.media_ok = false; A.media_msg = err; return; }
    A.mg.cancel = false;
    A.mg.progress = 0.f;
    A.mg.running = true;
    int count = A.gen_count;
    float secs = A.gen_seconds, temp = A.temperature;
    uint64_t seed = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
    A.mg.th = std::thread([snap, count, secs, temp, seed] {
        std::string e;
        auto m = Model::load(snap, &e);
        std::error_code ec;
        std::filesystem::remove(upath(snap), ec);
        Rng rng(seed);
        if (m && m->spec.media == MEDIA_IMAGE) {
            for (int k = 0; k < count && !A.mg.cancel; ++k) {
                Image img = sample_image(*m, temp, rng);
                std::lock_guard<std::mutex> lk(A.mg.mu);
                A.mg.images.push_back(std::move(img));
                A.mg.progress = (float)(k + 1) / count;
            }
        } else if (m && m->spec.media == MEDIA_AUDIO) {
            Sound snd = sample_sound(*m, secs, temp, rng, [](float done) {
                A.mg.progress = done;
                return !A.mg.cancel.load();
            });
            std::lock_guard<std::mutex> lk(A.mg.mu);
            A.mg.sound = std::move(snd);
        }
        A.mg.running = false;
    });
}

// Saves the generated pictures (one PNG sheet) or sound (WAV) under
// <config>/outputs and returns the path, or "" on failure.
std::string save_media(bool images) {
    std::filesystem::path dir = upath(config_dir()) / "outputs";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::string path = (dir / ((images ? "images-" : "sound-") + timestamp(true) + (images ? ".png" : ".wav"))).u8string(), err;
    std::lock_guard<std::mutex> lk(A.mg.mu);
    bool ok;
    if (images) {
        if (A.mg.images.empty()) return "";
        int s = A.mg.images[0].w, scale = std::max(1, 64 / std::max(1, s)), cell = s * scale + 2;
        int n = (int)A.mg.images.size(), cols = (int)std::ceil(std::sqrt((double)n)), rows = (n + cols - 1) / cols;
        Image sheet;
        sheet.w = cols * cell;
        sheet.h = rows * cell;
        sheet.rgb.assign((size_t)sheet.w * sheet.h * 3, 255);
        for (int k = 0; k < n; ++k)
            for (int y = 0; y < s * scale; ++y)
                for (int x = 0; x < s * scale; ++x)
                    for (int c = 0; c < 3; ++c)
                        sheet.rgb[((size_t)(k / cols * cell + 1 + y) * sheet.w + k % cols * cell + 1 + x) * 3 + c] =
                            A.mg.images[k].rgb[((size_t)(y / scale) * s + x / scale) * 3 + c];
        ok = save_png(path, sheet, &err);
    } else {
        ok = !A.mg.sound.samples.empty() && save_wav(path, A.mg.sound, &err);
    }
    return ok ? path : "";
}

void media_playground() {
    bool image = A.model->spec.media == MEDIA_IMAGE;
    begin_card("mgen", tr(image ? S::GenImages : S::GenSound), 0, 0,
               image ? strf(tr(S::ImageModelFmt), A.model->spec.img_size, A.model->spec.img_size, A.model->spec.vocab - 1).c_str()
                     : strf(tr(S::SoundModelFmt), A.model->spec.rate).c_str());
    float lw = std::min(px(200), ImGui::GetContentRegionAvail().x * 0.3f);
    float sw = std::min(px(320), ImGui::GetContentRegionAvail().x - lw - px(20));
    field_label(tr(S::Temperature), lw);
    ImGui::SetNextItemWidth(sw);
    slider_float("##mtemp", &A.temperature, 0.1f, 1.5f, "%.2f");
    if (image) {
        field_label(tr(S::HowMany), lw);
        ImGui::SetNextItemWidth(sw);
        slider_int("##mcount", &A.gen_count, 1, 36);
        A.gen_count = std::max(1, std::min(36, A.gen_count));
    } else {
        field_label(tr(S::Seconds), lw);
        ImGui::SetNextItemWidth(sw);
        slider_float("##msecs", &A.gen_seconds, 0.5f, 10.f, "%.1f s");
        A.gen_seconds = std::max(0.5f, std::min(10.f, A.gen_seconds));
    }
    gap(2);
    if (A.mg.running) {
        if (button(tr(S::Stop), BTN_DANGER, IC_STOP)) A.mg.cancel = true;
        ImGui::SameLine(0, px(16));
        ImGui::AlignTextToFramePadding();
        text(strf(tr(S::Generating), A.mg.progress.load() * 100).c_str(), C::muted);
    } else {
        if (button(tr(image ? S::GenImages : S::GenSound), BTN_PRIMARY, IC_SPARK)) start_media_gen();
        bool have;
        {
            std::lock_guard<std::mutex> lk(A.mg.mu);
            have = image ? !A.mg.images.empty() : !A.mg.sound.samples.empty();
        }
        if (have) {
            if (!image) {
                ImGui::SameLine();
                if (button(tr(S::Play), BTN_SECONDARY, IC_PLAY)) {
                    std::string p = (upath(config_dir()) / "playground.wav").u8string(), err;
                    std::lock_guard<std::mutex> lk(A.mg.mu);
                    if (save_wav(p, A.mg.sound, &err)) play_wav(p);
                }
            }
            ImGui::SameLine();
            if (button(tr(S::Save), BTN_SECONDARY, IC_SAVE)) {
                std::string p = save_media(image);
                A.media_ok = !p.empty();
                A.media_msg = p;
            }
            if (!A.media_msg.empty() && A.media_ok) {
                ImGui::SameLine();
                if (button(tr(S::OpenFolder), BTN_GHOST, IC_FOLDER)) open_folder(upath(A.media_msg).parent_path().u8string());
            }
        }
    }
    if (!A.media_msg.empty()) {
        if (A.media_ok) text_wrapped(strf(tr(S::SavedTo), A.media_msg.c_str()).c_str(), C::good, F.small);
        else text_wrapped(strf(tr(S::Failed), A.media_msg.c_str()).c_str(), C::bad, F.small);
    }
    end_card();
    gap();
    float rest = ImGui::GetContentRegionAvail().y;
    begin_card("mout", tr(S::Result));
    float inner = std::max(px(200), rest - ImGui::GetCursorPosY() - card_padding_y() - px(2));
    if (begin_inset("##mres", inner)) {
        std::lock_guard<std::mutex> lk(A.mg.mu);
        if (image) {
            while (A.gen_tex.size() < A.mg.images.size()) A.gen_tex.push_back(make_texture(A.mg.images[A.gen_tex.size()]));
            if (A.gen_tex.empty()) text(tr(S::GenImgEmpty), C::subtle);
            else image_grid(A.gen_tex, std::max(px(96), std::min(px(160), (ImGui::GetContentRegionAvail().x - px(40)) / 6)));
        } else if (A.mg.sound.samples.empty()) {
            text(tr(S::GenSndEmpty), C::subtle);
        } else {
            waveform(A.mg.sound.samples, ImVec2(ImGui::GetContentRegionAvail().x, std::min(px(200), inner - px(20))));
            text(strf("%.1f s, %d Hz", (double)A.mg.sound.samples.size() / A.mg.sound.rate, A.mg.sound.rate).c_str(), C::muted);
        }
    }
    ImGui::EndChild();
    end_card();
}

void page_test() {
    header(tr(S::NavTest), tr(S::TestHelp));
    header_end();
    if (!A.model) {
        int r = empty_state(IC_CHAT, nullptr, tr(S::NoModel), tr(S::NavTrain), tr(S::LoadModel));
        if (r == 1) A.page = PAGE_TRAIN;
        if (r == 2) A.page = PAGE_FILES;
        return;
    }
    if (A.model->spec.kind == MODEL_GPT && A.model->spec.media != MEDIA_TEXT) {
        media_playground();
        return;
    }
    if (A.model->spec.kind == MODEL_GPT) {
        begin_card("prompt", tr(S::Prompt));
        ImGui::InputTextMultiline("##prompt", A.prompt, sizeof A.prompt, ImVec2(-FLT_MIN, px(86)));
        float bw = std::max(button_width(tr(S::Generate), IC_SPARK), button_width(tr(S::Stop), IC_STOP));
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(px(12), 0));
        if (ImGui::BeginTable("##gen", 3, ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableSetupColumn("a", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("b", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("c", ImGuiTableColumnFlags_WidthFixed, bw);
            ImFont* sf = F.small ? F.small : ImGui::GetFont();
            float label_h = sf->FontSize + px(6);
            auto stacked_label = [&](const char* s) {
                ImVec2 lp = ImGui::GetCursorScreenPos();
                ImGui::Dummy(ImVec2(ImGui::GetContentRegionAvail().x, label_h));
                text_ellipsis(ImGui::GetWindowDrawList(), sf, lp, ImGui::GetContentRegionAvail().x, s, col(C::muted));
                ImGui::SetCursorScreenPos(ImVec2(lp.x, lp.y + label_h));
                ImGui::SetNextItemWidth(-FLT_MIN);
            };
            ImGui::TableNextColumn();
            stacked_label(tr(S::Temperature));
            slider_float("##temp", &A.temperature, 0.f, 1.5f, "%.2f");
            ImGui::TableNextColumn();
            stacked_label(tr(S::Length));
            slider_int("##len", &A.gen_len, 20, 2000);
            ImGui::TableNextColumn();
            ImGui::Dummy(ImVec2(1, label_h - ImGui::GetStyle().ItemSpacing.y));
            if (A.gen_left > 0) {
                if (button(tr(S::Stop), BTN_DANGER, IC_STOP, bw)) A.gen_left = 0;
            } else if (button(tr(S::Generate), BTN_PRIMARY, IC_SPARK, bw)) {
                A.gen_ids = A.model->encode(A.prompt);
                if (A.gen_ids.empty()) A.gen_ids.push_back(A.gen_rng.below(std::max(1, A.model->spec.vocab)));
                A.gen_out = A.prompt;
                A.gen_left = A.gen_len;
            }
            ImGui::EndTable();
        }
        ImGui::PopStyleVar();
        end_card();
        pump_generation();
        gap();
        float rest = ImGui::GetContentRegionAvail().y;
        begin_card("out", tr(S::Result));
        float inner = std::max(px(200), rest - ImGui::GetCursorPosY() - card_padding_y() - px(2));
        if (begin_inset("##out", inner)) {
            FontScope f(F.mono);
            if (A.gen_out.empty()) text(tr(S::GenEmpty), C::subtle);
            ImGui::PushTextWrapPos(0);
            ImGui::TextUnformatted(A.gen_out.data(), A.gen_out.data() + A.gen_out.size());
            ImGui::PopTextWrapPos();
            if (A.gen_left > 0) ImGui::SetScrollHereY(1.f);
        }
        ImGui::EndChild();
        end_card();
        return;
    }

    const ModelSpec& s = A.model->spec;
    if ((int)A.feat_in.size() != s.in_dim) {
        A.feat_in.assign(s.in_dim, 0.f);
        for (int j = 0; j < s.in_dim && j < (int)A.model->norm_mean.size(); ++j) A.feat_in[j] = A.model->norm_mean[j];
        A.pred.clear();
    }
    begin_card("inputs", tr(S::InputsLbl));
    int cols = ImGui::GetContentRegionAvail().x > px(700) ? 2 : 1;
    if (ImGui::BeginTable("##feat", cols, ImGuiTableFlags_SizingStretchSame)) {
        for (int j = 0; j < s.in_dim; ++j) {
            ImGui::TableNextColumn();
            ImGui::PushID(j);
            std::string name = j < (int)A.model->feat_names.size() ? A.model->feat_names[j] : "x" + std::to_string(j + 1);
            field_label(name.c_str(), std::min(px(200), ImGui::GetContentRegionAvail().x * 0.45f));
            ImGui::InputFloat("##v", &A.feat_in[j], 0, 0, "%.4f");
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    gap(2);
    if (button(tr(S::Predict), BTN_PRIMARY, IC_SPARK)) {
        std::lock_guard<std::mutex> lk(A.trainer.model_mu);
        A.pred = A.model->predict(A.feat_in);
    }
    end_card();
    if (A.pred.empty()) return;
    gap();
    begin_card("pred", tr(S::Result));
    if (s.regression) {
        FontScope f(F.stat);
        text(strf("%.5g", A.pred[0]).c_str(), C::accent_hi);
    } else {
        size_t best = 0;
        for (size_t i = 1; i < A.pred.size(); ++i)
            if (A.pred[i] > A.pred[best]) best = i;
        float aw = ImGui::GetContentRegionAvail().x;
        float nw = std::min(px(180), aw * 0.3f);
        ImFont* bf = F.bold ? F.bold : ImGui::GetFont();
        float pw = text_size(bf, "100.0%").x;
        float rh = ImGui::GetFontSize() + px(10);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        for (size_t i = 0; i < A.pred.size(); ++i) {
            std::string name = i < A.model->vocab.size() ? A.model->vocab[i] : std::to_string(i);
            ImVec2 p = ImGui::GetCursorScreenPos();
            ImGui::Dummy(ImVec2(aw, rh));
            bool top = i == best;
            ImFont* f = top ? bf : ImGui::GetFont();
            float ty = p.y + (rh - f->FontSize) * 0.5f;
            text_ellipsis(dl, f, ImVec2(p.x, ty), nw - px(12), name.c_str(), col(top ? C::text : C::muted));
            float x0 = p.x + nw, x1 = p.x + aw - pw - px(14), cy = p.y + rh * 0.5f;
            dl->AddRectFilled(ImVec2(x0, cy - px(4)), ImVec2(x1, cy + px(4)), col(C::surface3), px(4));
            if (A.pred[i] > 0)
                dl->AddRectFilled(ImVec2(x0, cy - px(4)), ImVec2(x0 + std::max(px(8), (x1 - x0) * A.pred[i]), cy + px(4)),
                                  col(top ? C::accent : C::border_hi), px(4));
            std::string pc = strf("%.1f%%", A.pred[i] * 100);
            float tw = text_size(f, pc.c_str()).x;
            dl->AddText(f, f->FontSize, ImVec2(p.x + aw - tw, ty), col(top ? C::accent_hi : C::muted), pc.c_str());
        }
    }
    end_card();
}

// ------------------------------------------------------------------ files

void page_files() {
    header(tr(S::NavFiles), tr(S::FilesHelp));
    header_end();
    begin_card("mpath", tr(S::ModelFile));
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##mpath", tr(S::PathHint), A.model_path, sizeof A.model_path);
    end_card();
    gap();

    static EqualRow row;
    float g = px(16), total = ImGui::GetContentRegionAvail().x;
    bool two = total > px(700);
    float cw = two ? std::floor((total - g) * 0.5f) : 0;
    float fh = ImGui::GetFrameHeight();

    begin_card("save", tr(S::SaveModel), two ? row.prev : 0, cw);
    text_wrapped(tr(S::SaveHelp), C::muted);
    if (!A.model) text_wrapped(tr(S::NoModel), C::warn, F.small);
    ImGui::BeginDisabled(!A.model);
    checkbox(tr(S::Half), &A.half);
    row.cur = std::max(row.cur, push_to_bottom(two ? row.prev : 0, fh));
    bool go = button(tr(S::Save), BTN_PRIMARY, IC_SAVE);
    if (has_native_dialogs()) {
        ImGui::SameLine();
        if (button(tr(S::Browse), BTN_SECONDARY, IC_FOLDER)) {
            std::string p = save_file_dialog();
            if (!p.empty()) { std::snprintf(A.model_path, sizeof A.model_path, "%s", p.c_str()); go = true; }
        }
    }
    if (go && A.model && A.model_path[0]) {
        std::string err;
        std::lock_guard<std::mutex> lk(A.trainer.model_mu);
        A.file_ok = A.model->save(A.model_path, A.half, &err);
        A.file_msg = A.file_ok ? tr(S::Saved) : err;
    }
    ImGui::EndDisabled();
    end_card();

    if (two) ImGui::SameLine(0, g);
    else gap();
    begin_card("load", tr(S::LoadModel), two ? row.prev : 0, two ? total - cw - g : 0);
    text_wrapped(tr(S::LoadHelp), C::muted);
    if (busy()) text_wrapped(tr(S::Busy), C::warn, F.small);
    ImGui::BeginDisabled(busy());
    row.cur = std::max(row.cur, push_to_bottom(two ? row.prev : 0, fh));
    bool load = button(tr(S::Load), BTN_PRIMARY, IC_UPLOAD);
    if (has_native_dialogs()) {
        ImGui::SameLine();
        if (button(tr(S::Browse), BTN_SECONDARY, IC_FOLDER)) {
            std::string p = open_file_dialog(DLG_MODEL);
            if (!p.empty()) { std::snprintf(A.model_path, sizeof A.model_path, "%s", p.c_str()); load = true; }
        }
    }
    if (load && A.model_path[0] && !busy()) load_model_file(A.model_path);
    ImGui::EndDisabled();
    end_card();
    row.next();

    if (!A.file_msg.empty()) {
        gap();
        if (A.file_ok) banner(A.file_msg.c_str(), C::good, IC_CHECK);
        else banner(strf(tr(S::Failed), A.file_msg.c_str()).c_str(), C::bad, IC_WARN);
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
    for (size_t i = 0; i < A.plugins.size() && i < A.plugin_args.size(); ++i)
        A.saved_args[A.plugins[i].file] = A.plugin_args[i].data();
    A.plugins = scan_plugins(plugin_dirs());
    A.plugin_args.assign(A.plugins.size(), {});
    for (size_t i = 0; i < A.plugins.size(); ++i) {
        auto it = A.saved_args.find(A.plugins[i].file);
        const std::string& a = it != A.saved_args.end() ? it->second : A.plugins[i].args;
        std::snprintf(A.plugin_args[i].data(), A.plugin_args[i].size(), "%s", a.c_str());
    }
    A.plugins_scanned = true;
}

// Folder that holds the `kor` Python package (next to the program, or in the source tree).
std::string python_lib_dir() {
    namespace fs = std::filesystem;
    fs::path exe = upath(exe_dir());
    for (const fs::path& base : {exe / "python", exe / ".." / "python", exe / ".." / ".." / "python"}) {
        std::error_code ec;
        if (fs::exists(base / "kor" / "__init__.py", ec)) return fs::weakly_canonical(base, ec).u8string();
    }
    return (exe / "python").u8string();
}

std::string bundled_index() {
    namespace fs = std::filesystem;
    fs::path exe = upath(exe_dir());
    for (const fs::path& base : {exe / "community", exe / ".." / "community", exe / ".." / ".." / "community"}) {
        std::error_code ec;
        if (fs::exists(base / "index.json", ec)) return fs::weakly_canonical(base / "index.json", ec).u8string();
    }
    return "";
}

// Environment shared by tools, skills and the plugin installer.
EnvVars plugin_env() {
    namespace fs = std::filesystem;
#if defined(_WIN32)
    const char* cli_name = "kor-cli.exe";
    const char sep = ';';
#else
    const char* cli_name = "kor-cli";
    const char sep = ':';
#endif
    std::error_code ec;
    fs::path cli = upath(exe_dir()) / cli_name;
    std::string pypath = python_lib_dir();
    for (auto& d : plugin_dirs()) pypath += std::string(1, sep) + d;
    return {
        {"PYTHONPATH", pypath},
        {"KOR_PYLIB", python_lib_dir()},
        {"PYTHONIOENCODING", "utf-8"},
        {"PYTHONUNBUFFERED", "1"},
        {"KOR_LANG", lang_code()},
        {"KOR_CLI", fs::exists(cli, ec) ? cli.u8string() : ""},
        {"KOR_PLUGINS_DIR", user_plugins_dir()},
        {"KOR_INDEX_FALLBACK", bundled_index()},
    };
}

// Runs `python -m kor <args>` (install, search, new) through the plugin runner.
void run_hub(int kind, const std::vector<std::string>& args) {
    if (A.runner.running() || !A.python.found()) return;
    std::vector<std::string> cmd = A.python.command();
    for (const char* a : {"-X", "utf8", "-m", "kor"}) cmd.push_back(a);
    cmd.insert(cmd.end(), args.begin(), args.end());
    if (!A.runner.start(cmd, plugin_env())) return;
    A.run_kind = kind;
    A.plugin_running = -1;
    if (kind == State::RUN_INDEX) {
        A.community_state = 1;
    } else {
        A.runner.clear_output();
        A.plugin_msg = tr(kind == State::RUN_INSTALL ? S::InstallRunning : S::PluginRunning);
        if (kind == State::RUN_NEW) A.plugin_msg = strf(tr(S::PluginRunning), tr(S::NewSkill));
        A.plugin_ok = true;
    }
}

void parse_community(const std::string& out) {
    A.community.clear();
    A.community_state = 3;
    size_t at = out.find('[');
    if (at == std::string::npos) return;
    json::Value v;
    if (!json::parse(out.substr(at, out.find_last_of(']') + 1 - at), v) || v.type != json::Value::ARR) return;
    static const char* const suffix[LANG_COUNT] = {"", ".tr", ".ru"};
    for (auto& e : v.a) {
        State::CommunityItem it;
        it.id = e["id"].str();
        it.type = e["type"].str();
        it.author = e["author"].str();
        it.url = e["url"].str();
        for (int l = 0; l < LANG_COUNT; ++l) {
            it.name[l] = e[(std::string("name") + suffix[l]).c_str()].str();
            it.desc[l] = e[(std::string("description") + suffix[l]).c_str()].str();
        }
        if (it.name[0].empty()) it.name[0] = it.id;
        if (!it.url.empty()) A.community.push_back(it);
    }
    A.community_state = 2;
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
    EnvVars env = plugin_env();
    env.push_back({"KOR_OUT", A.plugin_out_dir});
    env.push_back({"KOR_DATA", A.data.kind != DATA_NONE ? A.data.path : ""});
    env.push_back({"KOR_MODEL", model_file});
    env.push_back({"KOR_TESTS", tests_file});
    std::vector<std::string> cmd = python_launch(A.python.command(), p.path);
    for (auto& a : split_args(A.plugin_args[i].data())) cmd.push_back(a);
    if (A.runner.start(cmd, env)) {
        A.run_kind = State::RUN_TOOL;
        A.plugin_running = i;
        A.plugin_msg = strf(tr(S::PluginRunning), p.title(get_language()).c_str());
        A.plugin_ok = true;
    }
}

// Picks up a finished plugin run and loads what it produced; called every frame.
void poll_plugin() {
    namespace fs = std::filesystem;
    int code = 0;
    if (!A.runner.take_finished(code)) return;
    if (A.run_kind == State::RUN_INDEX) {
        if (code == 0) parse_community(A.runner.output());
        else A.community_state = 3;
        if (A.community_state == 2) A.runner.clear_output();
        return;
    }
    if (A.run_kind == State::RUN_INSTALL || A.run_kind == State::RUN_NEW) {
        A.plugin_ok = code == 0;
        if (A.run_kind == State::RUN_INSTALL) {
            A.plugin_msg = tr(code == 0 ? S::InstallDone : S::InstallFailed);
            if (code == 0) A.install_src[0] = '\0';
        } else if (code == 0) {
            std::string out = A.runner.output();
            while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
            std::string path = out.substr(out.find_last_of('\n') + 1);
            A.plugin_msg = strf(tr(S::NewSkillDone), path.c_str());
            open_folder(user_plugins_dir());
        } else {
            A.plugin_msg = strf(tr(S::PluginFailed), tr(S::NewSkill), code);
        }
        if (code == 0) scan_all_plugins();
        return;
    }
    int i = A.plugin_running;
    A.plugin_running = -1;
    std::string name = i >= 0 && i < (int)A.plugins.size() ? A.plugins[i].title(get_language()) : "?";
    A.plugin_ok = code == 0;
    if (code == -2) A.plugin_msg = strf(tr(S::PluginCancelled), name.c_str());
    else if (code == -1) A.plugin_msg = tr(S::PluginNoStart);
    else if (code != 0) A.plugin_msg = strf(tr(S::PluginFailed), name.c_str(), code);
    else A.plugin_msg = strf(tr(S::PluginDone), name.c_str());
    if (code != 0) return;

    fs::path out = upath(A.plugin_out_dir);
    std::error_code ec;
    for (const char* f : {"data.txt", "data.csv", "data.tsv"}) {
        fs::path dp = out / f;
        if (!fs::exists(dp, ec)) continue;
        int kind = std::strcmp(f, "data.txt") == 0 ? 0 : 1;
        if (load_data_impl(dp.u8string(), kind, -1)) {
            A.plugin_msg += "\n" + strf(tr(S::PluginData), dp.u8string().c_str());
        } else {
            A.plugin_msg += "\n" + strf(tr(S::Failed), A.data_msg.c_str());
            A.plugin_ok = false;
        }
        break;
    }
    std::string tests;
    if (read_file_text(out / "tests.txt", tests)) {
        set_tests(tests);
        A.results.clear();
        A.plugin_msg += "\n" + strf(tr(S::PluginTests), (int)current_tests().size());
    }
}

// One plugin card. Skills get an on/off switch, tools a Run button.
void plugin_card(int i, float width, EqualRow& row, bool can_run) {
    const Plugin& p = A.plugins[i];
    int lang = get_language();
    float fh = ImGui::GetFrameHeight();
    ImGui::PushID(i);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(px(10), px(10)));
    begin_card("plugin", nullptr, row.prev, width);
    float x0 = ImGui::GetCursorPosX(), cw = ImGui::GetContentRegionAvail().x, y0 = ImGui::GetCursorPosY();
    {
        FontScope f(F.bold);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(p.title(lang).c_str());
    }
    float by = y0 + (fh - (F.small_bold ? F.small_bold->FontSize : 13.f) - px(8)) * 0.5f;
    ImGui::SameLine(0, px(10));
    ImGui::SetCursorPosY(by);
    badge(p.file.c_str(), C::muted, false);
    if (!p.version.empty()) {
        ImGui::SameLine(0, px(6));
        ImGui::SetCursorPosY(by);
        badge(("v" + p.version).c_str(), C::blue, false);
    }
    if (p.skill) {
        bool on = A.skills_on.count(p.file) > 0;
        ImGui::SameLine(x0 + cw - toggle_width());
        ImGui::SetCursorPosY(y0);
        if (toggle("##on", &on)) {
            if (on) A.skills_on.insert(p.file);
            else A.skills_on.erase(p.file);
        }
        tooltip(tr(on ? S::SkillOn : S::SkillOff));
    }
    if (!p.description(lang).empty()) text_wrapped(p.description(lang).c_str(), C::muted);
    std::string meta;
    if (!p.author.empty()) meta = p.author;
    if (!p.packages.empty()) meta += (meta.empty() ? "" : "   \xC2\xB7   ") + strf(tr(S::NeedsPackages), p.packages.c_str());
    if (!meta.empty()) text_wrapped(meta.c_str(), C::subtle, F.small);
    row.cur = std::max(row.cur, push_to_bottom(row.prev, fh));
    if (p.skill) {
        ImGui::BeginDisabled(busy());
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::InputTextWithHint("##args", tr(S::Arguments), A.plugin_args[i].data(), A.plugin_args[i].size());
        ImGui::EndDisabled();
    } else {
        float bw = std::max(button_width(tr(S::Run), IC_PLAY), button_width(tr(S::Stop), IC_STOP));
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - bw - px(10));
        ImGui::InputTextWithHint("##args", tr(S::Arguments), A.plugin_args[i].data(), A.plugin_args[i].size());
        ImGui::SameLine();
        if (A.run_kind == State::RUN_TOOL && A.plugin_running == i && A.runner.running()) {
            if (button(tr(S::Stop), BTN_DANGER, IC_STOP, bw)) A.runner.cancel();
        } else {
            ImGui::BeginDisabled(!can_run);
            if (button(tr(S::Run), BTN_PRIMARY, IC_PLAY, bw)) run_plugin(i);
            ImGui::EndDisabled();
        }
    }
    end_card();
    ImGui::PopStyleVar();
    ImGui::PopID();
}

void plugin_section(const char* title, const char* help) {
    gap(4);
    {
        FontScope f(F.h2);
        ImGui::TextUnformatted(title);
    }
    text_wrapped(help, C::muted, F.small);
    gap(2);
}

void plugin_grid(const std::vector<int>& items, std::vector<EqualRow>& rows, bool can_run) {
    float g = px(16);
    int cols = fit_columns(px(440), 2, g);
    float cw = column_width(cols, g);
    rows.resize(items.size() / std::max(1, cols) + 2);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(g, g));
    for (int k = 0; k < (int)items.size(); ++k) {
        EqualRow& row = rows[k / cols];
        if (k % cols) ImGui::SameLine(0, g);
        plugin_card(items[k], cw, row, can_run);
        if (k % cols == cols - 1 || k + 1 == (int)items.size()) row.next();
    }
    ImGui::PopStyleVar();
}

bool plugin_installed(const std::string& url) {
    std::string file = url.substr(url.find_last_of('/') + 1);
    for (auto& p : A.plugins)
        if (p.file == file) return true;
    return false;
}

void share_card() {
    bool can = A.python.found() && !A.runner.running();
    float sp = ImGui::GetStyle().ItemSpacing.x;
    begin_card("share", tr(S::ShareTitle));
    text_wrapped(tr(S::ShareHelp), C::muted);
    float bw = button_width(tr(S::Install), IC_PLUS);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - bw - sp);
    bool enter = ImGui::InputTextWithHint("##src", tr(S::InstallHint), A.install_src, sizeof A.install_src, ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    ImGui::BeginDisabled(!can || !A.install_src[0]);
    if ((button(tr(S::Install), BTN_PRIMARY, IC_PLUS, bw) || enter) && can && A.install_src[0])
        run_hub(State::RUN_INSTALL, {"install", A.install_src, "--yes", "--dir", user_plugins_dir()});
    ImGui::EndDisabled();
    text_wrapped(tr(S::TrustWarn), C::warn, F.small);
    gap(2);
    ImGui::BeginDisabled(!can);
    if (button(tr(S::Community), BTN_SECONDARY, IC_GLOBE)) run_hub(State::RUN_INDEX, {"search", "--json"});
    ImGui::EndDisabled();
    if (A.community_state == 1) {
        ImGui::SameLine(0, px(12));
        ImGui::AlignTextToFramePadding();
        text(tr(S::CommunityLoading), C::muted);
    } else if (A.community_state == 3) {
        ImGui::SameLine(0, px(12));
        ImGui::AlignTextToFramePadding();
        text(tr(S::CommunityEmpty), C::bad);
    }
    if (A.community_state == 2 && !A.community.empty()) {
        int lang = get_language();
        ImGuiTableFlags tf = ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp;
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(px(8), px(10)));
        if (ImGui::BeginTable("##community", 3, tf)) {
            float iw = std::max(button_width(tr(S::Install), IC_PLUS), badge_width(tr(S::Installed)));
            float tw = std::max(badge_width("skill"), badge_width("tool"));
            ImGui::TableSetupColumn("##n", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("##t", ImGuiTableColumnFlags_WidthFixed, tw);
            ImGui::TableSetupColumn("##i", ImGuiTableColumnFlags_WidthFixed, iw);
            for (int k = 0; k < (int)A.community.size(); ++k) {
                const State::CommunityItem& it = A.community[k];
                ImGui::PushID(k);
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                {
                    FontScope f(F.bold);
                    ImGui::TextUnformatted(!it.name[lang].empty() ? it.name[lang].c_str() : it.name[0].c_str());
                }
                const std::string& d = !it.desc[lang].empty() ? it.desc[lang] : it.desc[0];
                std::string sub = d + (it.author.empty() ? "" : "  (" + it.author + ")");
                if (!sub.empty()) text_wrapped(sub.c_str(), C::muted, F.small);
                ImGui::TableNextColumn();
                bool skill = it.type == "skill";
                badge(skill ? "skill" : "tool", skill ? C::violet : C::teal, false);
                ImGui::TableNextColumn();
                if (plugin_installed(it.url)) {
                    badge(tr(S::Installed), C::good, true);
                } else {
                    ImGui::BeginDisabled(!can);
                    if (button(tr(S::Install), BTN_SECONDARY, IC_PLUS, iw))
                        run_hub(State::RUN_INSTALL, {"install", it.url, "--yes", "--dir", user_plugins_dir()});
                    ImGui::EndDisabled();
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
        ImGui::PopStyleVar();
    }
    end_card();
}

void page_plugins() {
    if (!A.plugins_scanned) {
        scan_all_plugins();
        A.python.start();
    }
    float sp = ImGui::GetStyle().ItemSpacing.x;
    header(tr(S::NavPlugins), tr(S::PluginsHelp),
           button_width(tr(S::NewSkill), IC_PLUS) + sp + button_width(tr(S::OpenPlugins), IC_FOLDER) + sp + button_width(tr(S::RefreshList), IC_REFRESH));
    ImGui::BeginDisabled(!A.python.found() || A.runner.running());
    if (button(tr(S::NewSkill), BTN_SECONDARY, IC_PLUS)) run_hub(State::RUN_NEW, {"new", "My skill", "--dir", user_plugins_dir()});
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (button(tr(S::OpenPlugins), BTN_SECONDARY, IC_FOLDER)) {
        std::error_code ec;
        std::filesystem::create_directories(upath(user_plugins_dir()), ec);
        open_folder(user_plugins_dir());
    }
    ImGui::SameLine();
    if (button(tr(S::RefreshList), BTN_GHOST, IC_REFRESH)) scan_all_plugins();
    header_end();

    if (A.python.searching()) banner(tr(S::PythonChecking), C::muted, IC_INFO);
    else if (A.python.found()) banner(strf(tr(S::PythonFound), A.python.version().c_str()).c_str(), C::good, IC_CHECK);
    else {
        banner(tr(S::PythonMissing), C::bad, IC_WARN);
        if (button(tr(S::CheckAgain), BTN_SECONDARY, IC_REFRESH)) A.python.start();
    }
    std::vector<std::string> dirs = plugin_dirs();
    {
        ImVec2 p = ImGui::GetCursorScreenPos();
        std::string s = strf(tr(S::PluginFolders), (int)dirs.size());
        ImFont* sf = F.small ? F.small : ImGui::GetFont();
        float tw = text_size(sf, s.c_str()).x;
        ImGui::Dummy(ImVec2(tw + px(22), sf->FontSize));
        bool hov = ImGui::IsItemHovered();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddText(sf, sf->FontSize, p, col(hov ? C::text : C::muted), s.c_str());
        icon(dl, IC_INFO, ImVec2(p.x + tw + px(12), p.y + sf->FontSize * 0.5f), px(13), col(hov ? C::text : C::subtle));
        std::string all;
        for (auto& d : dirs) all += (all.empty() ? "" : "\n") + d;
        tooltip(all.c_str());
    }
    bool can_run = A.python.found() && !busy() && !A.runner.running();
    std::vector<int> skills, tools;
    for (int i = 0; i < (int)A.plugins.size(); ++i) (A.plugins[i].skill ? skills : tools).push_back(i);
    static std::vector<EqualRow> skill_rows, tool_rows;

    plugin_section(tr(S::SkillsTitle), tr(S::SkillsHelp));
    if (skills.empty()) text_wrapped(tr(S::NoSkills), C::muted);
    else plugin_grid(skills, skill_rows, can_run);

    plugin_section(tr(S::ToolsTitle), tr(S::ToolsHelp));
    if (busy()) banner(tr(S::PluginBusy), C::warn, IC_INFO);
    if (tools.empty()) text_wrapped(tr(S::NoPlugins), C::muted);
    else plugin_grid(tools, tool_rows, can_run);

    gap(4);
    share_card();

    bool show_out = A.run_kind != State::RUN_INDEX;
    if (!A.plugin_msg.empty() && show_out) {
        gap();
        bool going = A.runner.running();
        banner(A.plugin_msg.c_str(), going ? C::accent : A.plugin_ok ? C::good : C::bad, going ? IC_CLOCK : A.plugin_ok ? IC_CHECK : IC_WARN);
    }
    std::string out = A.runner.output();
    if (out.empty() || !show_out) return;
    gap();
    float rest = ImGui::GetContentRegionAvail().y;
    begin_card("pout");
    {
        FontScope f(F.bold);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(tr(S::PluginOutput));
    }
    ImGui::SameLine(ImGui::GetWindowWidth() - px(20) - button_width(tr(S::Clear), IC_X));
    if (button(tr(S::Clear), BTN_GHOST, IC_X)) A.runner.clear_output();
    float inner = std::max(px(180), rest - ImGui::GetCursorPosY() - card_padding_y() - px(2));
    if (begin_inset("##pout", inner, true)) {
        FontScope f(F.mono);
        bool at_end = ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4;
        ImGui::TextUnformatted(out.data(), out.data() + out.size());
        if (at_end) ImGui::SetScrollHereY(1.f);
    }
    ImGui::EndChild();
    end_card();
}

// ---------------------------------------------------------------- sidebar

void sidebar(ImVec2 pos, float w, float h) {
    ImGuiWindowFlags wf = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                          ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollWithMouse;
    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize(ImVec2(w, h));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, C::sidebar);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(px(14), px(18)));
    ImGui::Begin("##sidebar", nullptr, wf);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->PushClipRectFullScreen();
    dl->AddLine(ImVec2(pos.x + w - 0.5f, pos.y), ImVec2(pos.x + w - 0.5f, pos.y + h), col(C::border));
    dl->PopClipRect();

    // brand
    ImVec2 p = ImGui::GetCursorScreenPos();
    float lz = px(36);
    draw_logo(dl, ImVec2(p.x + px(4), p.y), lz);
    ImFont* bf = F.bold ? F.bold : ImGui::GetFont();
    ImFont* sf = F.small ? F.small : ImGui::GetFont();
    float tx = p.x + px(4) + lz + px(12);
    float ty = p.y + (lz - bf->FontSize - sf->FontSize - px(1)) * 0.5f;
    dl->AddText(bf, bf->FontSize, ImVec2(tx, ty), col(C::text), "Kor Studio");
    dl->AddText(sf, sf->FontSize, ImVec2(tx, ty + bf->FontSize + px(1)), col(C::subtle), "v" KOR_VERSION);
    ImGui::Dummy(ImVec2(0, lz));
    gap(14);

    {
        ImFont* lf = F.small_bold ? F.small_bold : ImGui::GetFont();
        ImVec2 q = ImGui::GetCursorScreenPos();
        dl->AddText(lf, lf->FontSize, ImVec2(q.x + px(10), q.y), col(C::subtle), tr(S::Workspace));
        ImGui::Dummy(ImVec2(0, lf->FontSize));
    }

    TrainStats st = A.trainer.stats();
    static const S nav[] = {S::NavHome, S::NavData, S::NavModel, S::NavTrain, S::NavTest, S::NavFiles, S::NavPlugins};
    static const Icon icons[] = {IC_HOME, IC_DATA, IC_MODEL, IC_TRAIN, IC_CHAT, IC_FILES, IC_PLUGIN};
    static_assert(sizeof(nav) / sizeof(nav[0]) == PAGE_COUNT, "one label per page");
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(px(8), px(3)));
    for (int i = 0; i < PAGE_COUNT; ++i) {
        int status = 0;
        if (i == PAGE_DATA && A.data.kind != DATA_NONE) status = 1;
        if (i == PAGE_MODEL && A.model) status = 1;
        if (i == PAGE_TRAIN) status = busy() ? 2 : st.state == TS_DONE ? 1 : 0;
        if (i == PAGE_PLUGINS && A.runner.running()) status = 2;
        ImGui::PushID(i);
        if (nav_item(tr(nav[i]), icons[i], A.page == i, status)) A.page = i;
        ImGui::PopID();
    }
    ImGui::PopStyleVar();

    // ---- bottom: training card and language
    float fh = ImGui::GetFrameHeight();
    bool show_train = st.total > 0 && st.state != TS_IDLE;
    float card_h = px(12) * 2 + (F.small_bold ? F.small_bold->FontSize : 13.f) + px(8) + px(4) + px(8) + sf->FontSize;
    float block = fh + px(14) + px(16) + (show_train ? card_h + px(14) : 0);
    float y = std::max(ImGui::GetCursorPosY() + px(10), ImGui::GetWindowHeight() - px(18) - block);
    ImGui::SetCursorPosY(y);
    if (show_train) {
        ImVec2 c = ImGui::GetCursorScreenPos();
        float cw = ImGui::GetContentRegionAvail().x;
        if (ImGui::InvisibleButton("##traincard", ImVec2(cw, card_h))) A.page = PAGE_TRAIN;
        bool hov = ImGui::IsItemHovered();
        if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        dl->AddRectFilled(c, ImVec2(c.x + cw, c.y + card_h), col(hov ? C::surface2 : C::surface), px(10));
        dl->AddRect(c, ImVec2(c.x + cw, c.y + card_h), col(C::border), px(10));
        ImFont* lf = F.small_bold ? F.small_bold : ImGui::GetFont();
        const ImVec4& sc_ = state_color(st.state);
        float x = c.x + px(12), yy = c.y + px(12);
        ImVec2 dc(x + px(3), yy + lf->FontSize * 0.5f);
        if (st.state == TS_RUNNING) {
            float k = 0.5f + 0.5f * std::sin((float)ImGui::GetTime() * 5.f);
            dl->AddCircleFilled(dc, px(3) + px(3) * k, col(sc_, 0.3f * (1.f - k)));
        }
        dl->AddCircleFilled(dc, px(3), col(sc_));
        dl->AddText(lf, lf->FontSize, ImVec2(x + px(12), yy), col(sc_), state_name(st.state));
        std::string pct = strf("%.0f%%", 100.0 * st.step / std::max(1, st.total));
        float pw = text_size(lf, pct.c_str()).x;
        dl->AddText(lf, lf->FontSize, ImVec2(c.x + cw - px(12) - pw, yy), col(C::text), pct.c_str());
        yy += lf->FontSize + px(8);
        float bw = cw - px(24), frac = (float)st.step / std::max(1, st.total);
        dl->AddRectFilled(ImVec2(x, yy), ImVec2(x + bw, yy + px(4)), col(C::surface3), px(2));
        dl->AddRectFilled(ImVec2(x, yy), ImVec2(x + std::max(px(4), bw * frac), yy + px(4)), col(st.state == TS_DONE ? C::good : C::accent), px(2));
        yy += px(4) + px(8);
        std::string info = strf("%s %.4f", tr(S::Loss), st.loss);
        if (st.state == TS_RUNNING) info += "   \xC2\xB7   " + fmt_time(st.eta);
        text_ellipsis(dl, sf, ImVec2(x, yy), bw, info.c_str(), col(C::muted));
        gap(4);
    }
    {
        ImVec2 q = ImGui::GetCursorScreenPos();
        float lw = ImGui::GetContentRegionAvail().x;
        dl->AddLine(ImVec2(q.x - px(14), q.y), ImVec2(q.x + lw + px(14), q.y), col(C::border));
        ImGui::Dummy(ImVec2(0, px(4)));
    }
    {
        ImVec2 q = ImGui::GetCursorScreenPos();
        icon(dl, IC_GLOBE, ImVec2(q.x + px(10), q.y + fh * 0.5f), px(16), col(C::muted));
        ImGui::Dummy(ImVec2(px(24), fh));
        ImGui::SameLine(0, px(6));
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (begin_combo("##lang", language_name(get_language()))) {
            for (int l = 0; l < LANG_COUNT; ++l)
                if (ImGui::Selectable(language_name(l), l == get_language())) set_language(l);
            ImGui::EndCombo();
        }
    }
    ImGui::End();
}

}  // namespace

// -------------------------------------------------------------------- API

void init(float ui_scale) {
    load_fonts(ui_scale);
    apply_style();
    A.sys = detect_system();
    A.cfg.threads = recommended_threads(A.sys);
    int lang = system_language();
    std::ifstream in(upath(settings_file()), std::ios::binary);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t sp = line.find(' ');
        std::string key = line.substr(0, sp), value = sp == std::string::npos ? "" : line.substr(sp + 1);
        if (key == "lang") lang = std::atoi(value.c_str());
        else if (key == "skill" && !value.empty()) A.skills_on.insert(value);
        else if (key == "args") {
            size_t tab = value.find('\t');
            if (tab != std::string::npos) A.saved_args[value.substr(0, tab)] = value.substr(tab + 1);
        }
    }
    set_language(lang);
    scan_all_plugins();
    A.python.start();
}

void shutdown() {
    A.runner.cancel();
    A.trainer.stop();
    stop_media_gen();
    stop_sound();
    if (!A.save_settings) return;
    for (size_t i = 0; i < A.plugins.size() && i < A.plugin_args.size(); ++i)
        A.saved_args[A.plugins[i].file] = A.plugin_args[i].data();
    std::ofstream out(upath(settings_file()), std::ios::binary);
    out << "lang " << get_language() << "\n";
    for (auto& s : A.skills_on) out << "skill " << s << "\n";
    for (auto& kv : A.saved_args) {
        auto p = std::find_if(A.plugins.begin(), A.plugins.end(), [&](const Plugin& x) { return x.file == kv.first; });
        if (p != A.plugins.end() && p->args == kv.second) continue;  // still the default
        out << "args " << kv.first << "\t" << kv.second << "\n";
    }
}

void set_page(int page) {
    if (page >= 0 && page < PAGE_COUNT) A.page = page;
}

bool load_data(const char* path) {
    std::string p = path;
    std::string ext = p.size() > 4 ? p.substr(p.size() - 4) : "";
    std::error_code ec;
    int kind = ext == ".csv" || ext == ".tsv" ? 1 : 0;
    if (std::filesystem::is_directory(upath(p), ec)) {
        kind = 3;
        for (std::filesystem::recursive_directory_iterator it(upath(p), ec), end; !ec && it != end; it.increment(ec))
            if (is_image_file(it->path().u8string())) { kind = 2; break; }
    } else if (is_wav_file(p)) {
        kind = 3;
    }
    A.data_kind = kind;
    std::snprintf(A.data_path, sizeof A.data_path, "%s", p.c_str());
    return load_data_impl(p, kind, -1);
}

bool load_model(const char* path) { return load_model_file(path); }

void generate_media() {
    if (A.model && A.model->spec.kind == MODEL_GPT && A.model->spec.media != MEDIA_TEXT) start_media_gen();
}

void start_training() { begin_training(); }

void set_train_tab(int tab) { A.train_tab = std::max(0, std::min(3, tab)); }

void use_skills_for_session(const char* const* files, int n) {
    A.skills_on.clear();
    for (int i = 0; i < n; ++i) A.skills_on.insert(files[i]);
}

void keep_settings() { A.save_settings = false; }

void scroll_to_end() { A.scroll_end = true; }

void load_community_list() { A.want_community = true; }

void frame() {
    pump_events();
    poll_plugin();
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    float sbw = px(236);
    sidebar(vp->WorkPos, sbw, vp->WorkSize.y);

    ImGuiWindowFlags wf = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                          ImGuiWindowFlags_NoBringToFrontOnFocus;
    wf &= ~ImGuiWindowFlags_NoScrollbar;
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + sbw, vp->WorkPos.y));
    ImGui::SetNextWindowSize(ImVec2(vp->WorkSize.x - sbw, vp->WorkSize.y));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(px(32), px(28)));
    ImGui::Begin("##content", nullptr, wf);
    ImGui::PopStyleVar();
    ImGui::PushID(A.page);
    switch (A.page) {
        case PAGE_HOME: page_home(); break;
        case PAGE_DATA: page_data(); break;
        case PAGE_MODEL: page_model(); break;
        case PAGE_TRAIN: page_train(); break;
        case PAGE_TEST: page_test(); break;
        case PAGE_FILES: page_files(); break;
        case PAGE_PLUGINS: page_plugins(); break;
    }
    ImGui::PopID();
    gap(4);
    if (A.page != PAGE_TEST) pump_generation();
    if (A.want_community && A.python.found() && !A.runner.running()) {
        A.want_community = false;
        run_hub(State::RUN_INDEX, {"search", "--json"});
    }
    if (A.scroll_end && ImGui::GetScrollMaxY() > 0) ImGui::SetScrollY(ImGui::GetScrollMaxY());
    ImGui::End();
}

}  // namespace app
