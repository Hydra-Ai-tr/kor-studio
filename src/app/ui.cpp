#include "ui.h"

#include "i18n.h"
#include "kit.h"
#include "platform.h"
#include "plugins.h"

#include "../core/kernels.h"
#include "../core/media.h"
#include "../core/sysinfo.h"
#include "../skills/datatools.h"
#include "../skills/json.h"
#include "../skills/skill.h"

#include "imgui.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif
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
    std::string data_note;  // what the last import or cleaning did
    bool want_clean = false;
    CleanOptions clean_opt;
    std::string clean_src, clean_out;
    CleanReport clean_rep;
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

    // projects: a folder with project.korproj, the model and (single-file) data
    struct ProjectInfo {
        std::string dir, name, desc, data, data_path, model;
        std::time_t updated = 0;
        double params = 0;
        int kind = -1;  // data kind, -1 without data
    };
    std::string project_dir, project_name, project_desc;  // the open project ("" = none)
    std::vector<std::string> recent_projects;
    std::vector<ProjectInfo> projects;
    bool projects_scanned = false;
    std::string project_msg;
    bool project_ok = true;
    bool want_new_project = false;
    char new_project_name[128] = "", new_project_desc[256] = "";
    std::string new_project_err;

    // window frame and shell around the pages
    bool custom_frame = false, maximized = false, sidebar_folded = false, want_settings = false;
    int window_cmd = 0;
    float title_h = 0;
    std::vector<ImVec4> title_hot;  // clickable parts of the title bar (x0, y0, x1, y1)
    size_t seen_activity = 0;       // activity entries already shown under the bell
    ImTextureID hero_tex = 0;
    int hero_w = 0, hero_h = 0;
    std::string user;

    // the latest release on GitHub, asked for on a worker thread
    struct UpdateInfo {
        std::atomic<int> state{0};     // 0 idle, 1 checking, 2 done, 3 failed
        std::string latest, url, err;  // written before state turns 2 or 3
    };
    std::shared_ptr<UpdateInfo> update = std::make_shared<UpdateInfo>();
    bool update_check = true, update_started = false, update_seen = false;
    bool use_gpu = false;  // experimental OpenCL matrix products during training

    // finished and stopped trainings (<config>/runs/*.json) for the comparison tab
    struct RunRecord {
        std::string file, name, state, arch;
        std::time_t time = 0;
        int steps = 0, total = 0, batch = 0, best_step = 0, tests_passed = -1, tests_total = 0;
        double elapsed = 0, params = 0, speed = 0;
        float loss = -1, val_loss = -1, best_val = -1, val_acc = -1, lr = 0;
        std::vector<int> train_steps, val_steps;
        std::vector<float> train_hist, val_hist;
        bool selected = false;
    };
    std::vector<RunRecord> runs;
    bool runs_scanned = false;

    // numbers for the home page tiles
    std::vector<std::string> recent_data;  // datasets loaded before, newest first
    int dataset_count = 0, train_runs = 0;
    double data_bytes = 0, disk_used_gb = 0, disk_total_gb = 0, disk_at = -100;

    // activity feed (kept in <config>/activity.log)
    enum ActKind { ACT_TRAIN_START = 0, ACT_TRAIN_DONE, ACT_TRAIN_STOP, ACT_RESUMED, ACT_MODEL_SAVED, ACT_DATA_LOADED,
                   ACT_PROJECT_SAVED, ACT_PROJECT_OPENED, ACT_COUNT };
    struct Act {
        std::time_t t = 0;
        int kind = 0;
        std::string arg;
    };
    std::vector<Act> activity;

    // live machine load for the home page
    Usage usage;
    double usage_at = -10;
    int model_count = 0;
    double models_counted_at = -100;

    // quick search (Ctrl+K)
    bool want_palette = false;
    char palette_q[128] = "";
    int palette_sel = 0;

    // crash-safe training: the run saves itself to <config>/autosave
    bool resume_avail = false, resume_warn = false;
    CheckpointInfo resume_info;
    std::string resume_data, resume_msg;

    // chat page: its own model, independent of training
    struct Chat {
        std::shared_ptr<Model> model;
        std::string path, name, msg;
        bool ok = true;
        char q_tag[128] = "Kullanıcı: ", a_tag[128] = "Asistan: ";
        struct Msg { bool user; std::string text; };
        std::vector<Msg> msgs;
        std::vector<std::string> queue;  // messages waiting for the model
        char input[4096] = "";
        float temperature = 0.3f;
        int max_len = 200;
        bool memory = true, settings_open = false;
        std::thread th;
        std::atomic<bool> running{false}, cancel{false};
        std::mutex mu;      // guards reply
        std::string reply;  // the answer the worker is writing
        Rng rng{77};
        bool focus = true, scroll = false;
        struct File { std::string path, name, when; double mb = 0; };
        std::vector<File> files;
        std::vector<std::string> recent;  // opened from elsewhere, newest first
        bool scanned = false;
    } chat;
};
State A;

bool busy() { return A.trainer.running(); }

// ---------------------------------------------------------------- updates

const char* const kReleaseApi = "https://api.github.com/repos/Hydra-Ai-tr/kor-studio/releases/latest";
const char* const kReleasePage = "https://github.com/Hydra-Ai-tr/kor-studio/releases/";

// True if version `a` ("v1.2.3" or "1.2.3") is newer than `b`.
bool newer_version(const std::string& a, const std::string& b) {
    auto parts = [](const std::string& v) {
        std::array<int, 3> p{0, 0, 0};
        size_t i = v.find_first_of("0123456789");
        for (int k = 0; k < 3 && i < v.size(); ++k) {
            p[k] = std::atoi(v.c_str() + i);
            i = v.find('.', i);
            if (i == std::string::npos) break;
            ++i;
        }
        return p;
    };
    return parts(a) > parts(b);
}

void start_update_check() {
    if (A.update->state == 1) return;
    auto u = std::make_shared<State::UpdateInfo>();
    u->state = 1;
    A.update = u;
    A.update_seen = false;
    std::thread([u] {
        std::string body, err;
        json::Value v;
        if (!http_get(kReleaseApi, body, err)) {
            u->err = err;
            u->state = 3;
        } else if (!json::parse(body, v) || v["tag_name"].str().empty()) {
            u->err = "unexpected reply";
            u->state = 3;
        } else {
            u->latest = v["tag_name"].str();
            const std::string& page = v["html_url"].str();
            // only ever open the project's own release pages
            u->url = page.rfind(kReleasePage, 0) == 0 ? page : std::string(kReleasePage) + "latest";
            u->state = 2;
        }
    }).detach();
}

bool update_ready() { return A.update->state == 2 && newer_version(A.update->latest, KOR_VERSION); }

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

// --------------------------------------------------------------- activity

std::string activity_file() { return (upath(config_dir()) / "activity.log").u8string(); }

void add_activity(int kind, std::string arg) {
    for (char& c : arg)
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    A.activity.push_back({std::time(nullptr), kind, arg});
    if (A.activity.size() > 50) A.activity.erase(A.activity.begin(), A.activity.end() - 50);
    if (!A.save_settings) return;  // screenshot runs leave no trace
    std::ofstream o(upath(activity_file()), std::ios::binary | std::ios::trunc);
    for (auto& a : A.activity) o << (long long)a.t << '\t' << a.kind << '\t' << a.arg << '\n';
}

void load_activity() {
    std::ifstream in(upath(activity_file()), std::ios::binary);
    std::string line;
    while (std::getline(in, line)) {
        size_t t1 = line.find('\t'), t2 = t1 == std::string::npos ? t1 : line.find('\t', t1 + 1);
        if (t2 == std::string::npos) continue;
        State::Act a;
        a.t = (std::time_t)std::atoll(line.substr(0, t1).c_str());
        a.kind = std::atoi(line.substr(t1 + 1, t2 - t1 - 1).c_str());
        a.arg = line.substr(t2 + 1);
        if (a.kind >= 0 && a.kind < State::ACT_COUNT && a.t > 0) A.activity.push_back(a);
    }
    if (A.activity.size() > 50) A.activity.erase(A.activity.begin(), A.activity.end() - 50);
}

std::string act_text(const State::Act& a) {
    static const S fmt[State::ACT_COUNT] = {S::ActTrainStart, S::ActTrainDone, S::ActTrainStop, S::ActResumed,
                                            S::ActModelSaved, S::ActDataLoaded, S::ActProjectSaved, S::ActProjectOpened};
    return strf(tr(fmt[a.kind]), a.arg.c_str());
}

std::string ago(std::time_t t) {
    double s = std::difftime(std::time(nullptr), t);
    if (s < 60) return tr(S::AgoNow);
    if (s < 3600) return strf(tr(S::AgoMin), (int)(s / 60));
    if (s < 86400) return strf(tr(S::AgoHour), (int)(s / 3600));
    return strf(tr(S::AgoDay), (int)(s / 86400));
}

std::string file_name(const std::string& path) { return path.empty() ? "" : upath(path).filename().u8string(); }

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

// ----------------------------------------------------- autosave and models

std::string autosave_path() { return (upath(config_dir()) / "autosave" / "training.kor").u8string(); }
std::string models_dir() { return (upath(config_dir()) / "models").u8string(); }

// What the app needs to continue a saved run besides the trainer's own
// state: where the data is and how it was read, then the question tests.
// Projects store the same, after their own keys.
std::string checkpoint_note(const std::string& data_path) {
    std::ostringstream o;
    o << "data=" << data_path << "\n"
      << "kind=" << A.data_kind << "\n"
      << "target=" << A.data.target_col << "\n"
      << "subword=" << (A.subword ? A.subword_size : 0) << "\n"
      << "img_size=" << A.img_size << "\nimg_colors=" << A.img_colors << "\nimg_mirror=" << (A.img_mirror ? 1 : 0) << "\n"
      << "snd_rate=" << A.snd_rate << "\n"
      << "from_file=" << (A.from_file ? 1 : 0) << "\n"
      << "stop_on_pass=" << (A.stop_on_pass ? 1 : 0) << "\n"
      << "---\n" << A.tests;
    return o.str();
}

std::map<std::string, std::string> parse_note(const std::string& note, std::string* tests) {
    std::map<std::string, std::string> kv;
    size_t pos = 0;
    while (pos < note.size()) {
        size_t nl = note.find('\n', pos);
        std::string line = note.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
        pos = nl == std::string::npos ? note.size() : nl + 1;
        if (line == "---") {
            if (tests) *tests = note.substr(pos);
            break;
        }
        size_t eq = line.find('=');
        if (eq != std::string::npos) kv[line.substr(0, eq)] = line.substr(eq + 1);
    }
    return kv;
}

// Looks for a saved unfinished run (at start and when a run stops).
void refresh_resume() {
    A.resume_avail = false;
    A.resume_data.clear();
    std::string ck = autosave_path();
    if (!has_checkpoint(ck)) return;
    std::string err;
    if (!read_checkpoint(ck, A.resume_info, &err) || A.resume_info.step <= 0) return;
    std::string data = parse_note(A.resume_info.note, nullptr)["data"];
    A.resume_data = upath(data).filename().u8string();
    A.resume_avail = true;
}

void save_open_project();

// A finished model goes to <config>/models, where the chat page finds it.
void save_finished_model() {
    if (!A.model) return;
    std::error_code ec;
    std::filesystem::create_directories(upath(models_dir()), ec);
    std::string stem = A.data.path.empty() ? "model" : upath(A.data.path).stem().u8string();
    if (stem.empty()) stem = "model";
    for (char& c : stem)
        if (c == ' ' || c == ':' || c == '/' || c == '\\') c = '_';
    std::string path = (upath(models_dir()) / (stem + "-" + timestamp(true) + ".kor")).u8string(), err;
    bool ok;
    {
        std::lock_guard<std::mutex> lk(A.trainer.model_mu);
        ok = A.model->save(path + ".tmp", false, &err);
    }
    if (ok) {
        std::filesystem::rename(upath(path + ".tmp"), upath(path), ec);
        ok = !ec;
        if (ec) err = ec.message();
    }
    if (ok) {
        log_fmt(State::T_GOOD, S::LogSavedModel, path.c_str());
        add_activity(State::ACT_MODEL_SAVED, file_name(path));
        A.chat.scanned = false;
        A.models_counted_at = -100;
    } else {
        std::filesystem::remove(upath(path + ".tmp"), ec);
        log_fmt(State::T_BAD, S::LogError, err.c_str());
    }
}

// Moves new training events into the log; called every frame.
// ------------------------------------------------------------- run records

std::string runs_dir() { return (upath(config_dir()) / "runs").u8string(); }

// Writes a summary and the curves of the run that just ended.
void save_run_record(const char* state) {
    TrainStats st = A.trainer.stats();
    if (st.step <= 0 || !A.save_settings || !A.model) return;
    json::Writer w;
    w.begin_object();
    w.field("name", file_name(A.data.path)).field("state", state).field("time", (long long)std::time(nullptr));
    w.field("steps", st.step).field("total", st.total).field("elapsed", st.elapsed).field("speed", st.items_per_sec);
    w.field("params", (double)A.model->param_count(false)).field("batch", A.cfg.batch).field("lr", (double)A.cfg.lr);
    const ModelSpec& s = A.model->spec;
    std::string arch = s.kind == MODEL_GPT ? strf("%d x %d, ctx %d", s.nlayer, s.dim, s.ctx) : strf("MLP %d", (int)s.layers.size());
    if (s.lora_rank > 0) arch += strf(", LoRA %d", s.lora_rank);
    w.field("arch", arch);
    w.field("loss", (double)st.loss).field("val_loss", (double)st.val_loss).field("best_val", (double)st.best_val);
    w.field("best_step", st.best_step).field("val_acc", (double)st.val_acc);
    w.field("tests_passed", st.probe_pass).field("tests_total", st.probe_total);
    // at most 400 points of the training curve
    int n = (int)st.train_hist.size(), stride = std::max(1, (n + 399) / 400);
    w.key("train").begin_array();
    for (int i = 0; i < n; i += stride) w.begin_array().value(i + 1).value((double)st.train_hist[i]).end_array();
    if (n > 0 && (n - 1) % stride) w.begin_array().value(n).value((double)st.train_hist[n - 1]).end_array();
    w.end_array();
    w.key("val").begin_array();
    for (size_t i = 0; i < st.val_hist.size() && i < st.val_steps.size(); ++i)
        w.begin_array().value(st.val_steps[i]).value((double)st.val_hist[i]).end_array();
    w.end_array();
    w.end_object();
    std::error_code ec;
    std::filesystem::create_directories(upath(runs_dir()), ec);
    char stamp[32];
    std::time_t now = std::time(nullptr);
    std::strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", std::localtime(&now));
    std::ofstream o(upath(runs_dir()) / (std::string(stamp) + ".json"), std::ios::binary);
    o << w.str();
    A.runs_scanned = false;
}

void scan_runs() {
    A.runs_scanned = true;
    std::set<std::string> selected;
    for (auto& r : A.runs)
        if (r.selected) selected.insert(r.file);
    A.runs.clear();
    std::error_code ec;
    for (std::filesystem::directory_iterator it(upath(runs_dir()), ec), end; !ec && it != end; it.increment(ec)) {
        if (it->path().extension() != ".json") continue;
        std::ifstream in(it->path(), std::ios::binary);
        std::stringstream ss;
        ss << in.rdbuf();
        json::Value v;
        if (!json::parse(ss.str(), v) || v.type != json::Value::OBJ) continue;
        State::RunRecord r;
        r.file = it->path().u8string();
        r.name = v["name"].str();
        r.state = v["state"].str();
        r.arch = v["arch"].str();
        r.time = (std::time_t)v["time"].num();
        r.steps = (int)v["steps"].num();
        r.total = (int)v["total"].num();
        r.batch = (int)v["batch"].num();
        r.best_step = (int)v["best_step"].num();
        r.tests_passed = (int)v["tests_passed"].num(-1);
        r.tests_total = (int)v["tests_total"].num();
        r.elapsed = v["elapsed"].num();
        r.params = v["params"].num();
        r.speed = v["speed"].num();
        r.loss = (float)v["loss"].num(-1);
        r.val_loss = (float)v["val_loss"].num(-1);
        r.best_val = (float)v["best_val"].num(-1);
        r.val_acc = (float)v["val_acc"].num(-1);
        r.lr = (float)v["lr"].num();
        for (const json::Value& p : v["train"].a)
            if (p.a.size() == 2) { r.train_steps.push_back((int)p.a[0].num()); r.train_hist.push_back((float)p.a[1].num()); }
        for (const json::Value& p : v["val"].a)
            if (p.a.size() == 2) { r.val_steps.push_back((int)p.a[0].num()); r.val_hist.push_back((float)p.a[1].num()); }
        r.selected = selected.count(r.file) > 0;
        A.runs.push_back(std::move(r));
    }
    std::sort(A.runs.begin(), A.runs.end(), [](const State::RunRecord& a, const State::RunRecord& b) { return a.time > b.time; });
    if (A.runs.size() > 100) A.runs.resize(100);
    if (selected.empty())
        for (size_t i = 0; i < A.runs.size() && i < 3; ++i) A.runs[i].selected = true;
}

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
            case EV_RESUMED:
                std::snprintf(b, sizeof b, tr(S::LogResumed), e.step);
                log_line(t + b, State::T_ACCENT);
                break;
            case EV_CHECKPOINT:
                std::snprintf(b, sizeof b, tr(S::LogCheckpointFail), e.step, e.text.c_str());
                log_line(t + b, State::T_BAD);
                break;
        }
        if (e.kind == EV_DONE) add_activity(State::ACT_TRAIN_DONE, file_name(A.data.path));
        if (e.kind == EV_STOPPED) add_activity(State::ACT_TRAIN_STOP, file_name(A.data.path));
        if (e.kind == EV_DONE || e.kind == EV_STOPPED) save_run_record(e.kind == EV_DONE ? "done" : "stopped");
        if (e.kind == EV_DONE && A.save_settings) {
            save_finished_model();
            if (!A.project_dir.empty()) save_open_project();
        }
        if (e.kind == EV_DONE || e.kind == EV_STOPPED || e.kind == EV_ERROR) {
            A.log_file.close();
            if (A.save_settings) refresh_resume();
        }
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

ImTextureID make_texture(const Image& img, bool smooth = false) {
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, smooth ? GL_LINEAR : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, smooth ? GL_LINEAR : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
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

bool read_whole(const std::string& path, std::string& out) {
    std::ifstream f(upath(path), std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// A dataset made from another one goes next to it, or into the settings folder
// when that folder cannot be written.
std::string write_derived(const std::string& src, const std::string& suffix, const std::string& text) {
    std::filesystem::path p = upath(src);
    std::string stem = p.stem().u8string(), tag = upath(suffix).stem().u8string();
    // cleaning a cleaned copy again replaces it instead of making "-clean-clean"
    if (stem.size() > tag.size() && stem.compare(stem.size() - tag.size(), tag.size(), tag) == 0) stem.resize(stem.size() - tag.size());
    std::filesystem::path name = upath(stem + suffix);
    for (const std::filesystem::path& dir : {p.parent_path(), upath(config_dir()) / "datasets"}) {
        std::error_code ec;
        if (!dir.empty()) std::filesystem::create_directories(dir, ec);
        std::ofstream o(dir / name, std::ios::binary);
        o << text;
        o.close();
        if (o) return (dir / name).u8string();
    }
    return "";
}

bool load_data_impl(const std::string& path, int kind, int target_col);

bool import_data(const std::string& path) {
    static const char* q_tags[LANG_COUNT] = {"User: ", "Kullanıcı: ", "Пользователь: "};
    static const char* a_tags[LANG_COUNT] = {"Assistant: ", "Asistan: ", "Ассистент: "};
    std::string src, out, err;
    ImportReport r;
    int l = get_language();
    if (!read_whole(path, src)) err = "cannot open file";
    else if (import_jsonl(src, q_tags[l], a_tags[l], out, r, &err)) {
        std::string dst = write_derived(path, "-import.txt", out);
        if (dst.empty()) err = strf(tr(S::CantWrite), (upath(path).stem().u8string() + "-import.txt").c_str());
        else if (load_data_impl(dst, 0, -1)) {
            A.data_note = strf(tr(S::ImportDone), (int)r.dialogues, r.format.c_str(), file_name(dst).c_str());
            return true;
        } else return false;
    }
    A.data_ok = false;
    A.data_msg = err;
    return false;
}

bool load_data_impl(const std::string& path, int kind, int target_col) {
    if (busy()) return false;
    std::string ext = upath(path).extension().u8string();
    for (char& ch : ext) ch = (char)std::tolower((unsigned char)ch);
    if (kind == 0 && (ext == ".jsonl" || ext == ".json")) return import_data(path);
    A.data_note.clear();
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
    add_activity(State::ACT_DATA_LOADED, file_name(path));
    auto& rd = A.recent_data;
    rd.erase(std::remove(rd.begin(), rd.end(), path), rd.end());
    rd.insert(rd.begin(), path);
    if (rd.size() > 30) rd.resize(30);
    A.models_counted_at = -100;
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

// Starts a run, or continues the saved one (`resume`; the model, data and
// settings must already be restored, see resume_training()).
void begin_training(bool resume = false) {
    if (busy() || A.data.kind == DATA_NONE) return;
    // Screenshot runs (settings left alone) neither save nor replace the autosave.
    bool autosave = A.save_settings;
    if (!resume && autosave && A.resume_avail) {
        A.resume_warn = true;
        A.page = PAGE_TRAIN;
        return;
    }
    if (!A.model || !model_matches_data()) {
        if (resume) return;
        build_model();
    }
    A.gen_left = 0;
    if (!resume) A.cfg.seed = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
    A.cfg.resume = resume && autosave;
    A.cfg.gpu = A.use_gpu;
    A.cfg.checkpoint.clear();
    if (autosave) {
        std::error_code ec;
        std::filesystem::create_directories(upath(autosave_path()).parent_path(), ec);
        A.cfg.checkpoint = autosave_path();
        A.cfg.checkpoint_every = 60;
        A.cfg.checkpoint_note = checkpoint_note(A.data.path);
    }
    if (A.model->spec.kind == MODEL_GPT && A.model->spec.media == MEDIA_TEXT) {
        // The chat page reads the dialogue tags from the saved model.
        A.model->meta.erase("q_tag");
        A.model->meta.erase("a_tag");
        if (!A.q_tag.empty() && !A.a_tag.empty()) {
            A.model->meta["q_tag"] = A.q_tag;
            A.model->meta["a_tag"] = A.a_tag;
        }
    }
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
        add_activity(resume ? State::ACT_RESUMED : State::ACT_TRAIN_START, file_name(A.data.path));
        if (autosave) {
            A.resume_avail = false;
            A.resume_warn = false;
            A.resume_msg.clear();
        }
    }
}

// Restores the data, model and settings of the saved run and continues it.
void resume_training() {
    if (busy()) return;
    A.resume_msg.clear();
    std::string ck = autosave_path(), err, tests;
    CheckpointInfo info;
    if (!read_checkpoint(ck, info, &err)) { A.resume_msg = err; return; }
    auto kv = parse_note(info.note, &tests);
    auto num = [&](const char* k, int def) { auto it = kv.find(k); return it == kv.end() ? def : std::atoi(it->second.c_str()); };
    std::string data = kv["data"];
    std::error_code ec;
    if (data.empty() || !std::filesystem::exists(upath(data), ec)) {
        A.resume_msg = strf(tr(S::ResumeNoData), data.c_str());
        return;
    }
    auto m = Model::load(ck, &err);
    if (!m) { A.resume_msg = err; return; }
    int kind = std::max(0, std::min(3, num("kind", 0)));
    int sub = num("subword", 0);
    A.subword = sub > 0;
    if (sub > 0) A.subword_size = sub;
    A.img_size = num("img_size", A.img_size);
    A.img_colors = num("img_colors", A.img_colors);
    A.img_mirror = num("img_mirror", 1) != 0;
    A.snd_rate = num("snd_rate", A.snd_rate);
    // With the saved model in place first, images and sound are read in its format.
    A.model = m;
    A.spec = m->spec;
    A.from_file = true;
    reset_generation();
    A.data_kind = kind;
    std::snprintf(A.data_path, sizeof A.data_path, "%s", data.c_str());
    if (!load_data_impl(data, kind, num("target", -1))) { A.resume_msg = A.data_msg; return; }
    if (A.model != m || !model_matches_data()) {
        A.resume_msg = "the data does not fit the saved model any more";
        return;
    }
    A.from_file = num("from_file", 0) != 0;
    set_tests(tests);
    A.stop_on_pass = num("stop_on_pass", 0) != 0;
    TrainConfig c = info.cfg;
    c.threads = A.cfg.threads;
    c.gpu = A.use_gpu;
    A.cfg = c;
    A.page = PAGE_TRAIN;
    begin_training(true);
}

void discard_resume() {
    remove_checkpoint(autosave_path());
    A.resume_avail = false;
    A.resume_warn = false;
    A.resume_msg.clear();
}

// ---------------------------------------------------------------- projects

const char* const kProjectFile = "project.korproj";

std::string projects_root() { return (upath(documents_dir()) / "Kor Studio Projects").u8string(); }

void remember_project(const std::string& dir) {
    auto& r = A.recent_projects;
    r.erase(std::remove(r.begin(), r.end(), dir), r.end());
    r.insert(r.begin(), dir);
    if (r.size() > 12) r.resize(12);
}

bool read_project(const std::string& dir, std::map<std::string, std::string>& kv, std::string& tests) {
    std::ifstream in(upath(dir) / kProjectFile, std::ios::binary);
    if (!in) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    kv = parse_note(ss.str(), &tests);
    return true;
}

// Moves `tmp` over `dst` (also where rename cannot replace a file).
bool move_over(const std::filesystem::path& tmp, const std::filesystem::path& dst, std::string& err) {
    std::error_code ec;
    std::filesystem::rename(tmp, dst, ec);
    if (ec) {
        std::filesystem::remove(dst, ec);
        std::filesystem::rename(tmp, dst, ec);
    }
    if (ec) err = ec.message();
    return !ec;
}

// Writes the data settings, question tests, training settings and the model
// into `dir`. Data that is a single file is copied in, so the folder is
// complete on its own; folders of pictures or sound are referenced.
bool save_project(const std::string& dir, const std::string& name, const std::string& desc, std::string& err) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path root = upath(dir);
    fs::create_directories(root, ec);
    if (ec) { err = ec.message(); return false; }
    std::map<std::string, std::string> old;
    std::string old_tests;
    read_project(dir, old, old_tests);
    std::string data = A.data.kind != DATA_NONE ? A.data.path : "";
    if (!data.empty()) {
        fs::path src = upath(data);
        if (fs::is_regular_file(src, ec) && fs::file_size(src, ec) < (200ull << 20)) {
            fs::path dst = root / "data" / src.filename();
            std::error_code eq;
            if (!fs::equivalent(src, dst, eq)) {
                fs::create_directories(dst.parent_path(), ec);
                fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
                if (ec) { err = ec.message(); return false; }
            }
            data = "data/" + src.filename().u8string();
        }
    }
    std::string model_file;
    long long params = 0;
    if (A.model) {
        fs::path mp = root / "model.kor";
        bool ok;
        {
            std::lock_guard<std::mutex> lk(A.trainer.model_mu);
            ok = A.model->save(mp.u8string() + ".tmp", false, &err);
            params = (long long)A.model->param_count(false);
        }
        if (!ok || !move_over(upath(mp.u8string() + ".tmp"), mp, err)) return false;
        model_file = "model.kor";
    }
    const TrainConfig& c = A.cfg;
    std::ostringstream o;
    o << "name=" << name << "\n"
      << "desc=" << desc << "\n"
      << "created=" << (old.count("created") ? old["created"] : std::to_string((long long)std::time(nullptr))) << "\n"
      << "updated=" << (long long)std::time(nullptr) << "\n"
      << "version=" << KOR_VERSION << "\n"
      << "model=" << model_file << "\n"
      << "params=" << params << "\n"
      << "cfg.steps=" << c.steps << "\ncfg.batch=" << c.batch << "\ncfg.accum=" << c.accum << "\ncfg.lr=" << c.lr
      << "\ncfg.wd=" << c.weight_decay << "\ncfg.eval_every=" << c.eval_every << "\ncfg.patience=" << c.patience
      << "\ncfg.optimizer=" << c.optimizer << "\ncfg.dropout=" << c.dropout << "\ncfg.prompt_weight=" << c.prompt_weight
      << "\ncfg.align=" << c.align << "\ncfg.ema=" << c.ema << "\ncfg.keep_best=" << (c.keep_best ? 1 : 0)
      << "\ncfg.probe_every=" << c.probe_every << "\n"
      << checkpoint_note(data);
    fs::path pf = root / kProjectFile, tmp = root / (std::string(kProjectFile) + ".tmp");
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out << o.str();
        if (!out) { err = "cannot write the project file"; return false; }
    }
    return move_over(tmp, pf, err);
}

void save_open_project() {
    if (A.project_dir.empty()) return;
    std::string err;
    A.project_ok = save_project(A.project_dir, A.project_name, A.project_desc, err);
    A.project_msg = A.project_ok ? tr(S::ProjectSaved) : err;
    A.projects_scanned = false;
    if (A.project_ok) add_activity(State::ACT_PROJECT_SAVED, A.project_name);
}

bool create_project(std::string name, std::string desc, std::string& err) {
    for (char& ch : desc)
        if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
    std::string clean;
    for (char ch : name)
        if (!std::strchr("/\\:*?\"<>|\t\r\n", ch)) clean += ch;
    while (!clean.empty() && (clean.back() == ' ' || clean.back() == '.')) clean.pop_back();
    while (!clean.empty() && clean.front() == ' ') clean.erase(clean.begin());
    if (clean.empty()) { err = tr(S::ProjectName); return false; }
    std::string dir = (upath(projects_root()) / upath(clean)).u8string();
    std::error_code ec;
    if (std::filesystem::exists(upath(dir) / kProjectFile, ec)) { err = tr(S::ProjectExists); return false; }
    if (!save_project(dir, clean, desc, err)) return false;
    A.project_dir = dir;
    A.project_name = clean;
    A.project_desc = desc;
    A.project_ok = true;
    A.project_msg = tr(S::ProjectSaved);
    remember_project(dir);
    A.projects_scanned = false;
    add_activity(State::ACT_PROJECT_SAVED, clean);
    return true;
}

// Opens a project folder (or its project.korproj). `err` may hold a warning
// even when it succeeds (e.g. the data was not found).
bool open_project_dir(std::string dir, std::string& err) {
    namespace fs = std::filesystem;
    if (busy()) { err = tr(S::Busy); return false; }
    std::error_code ec;
    fs::path p = upath(dir);
    if (fs::is_regular_file(p, ec)) p = p.parent_path();
    dir = p.u8string();
    std::map<std::string, std::string> kv;
    std::string tests;
    if (!read_project(dir, kv, tests)) { err = std::string(kProjectFile) + " ?"; return false; }
    auto num = [&](const char* k, double def) { auto it = kv.find(k); return it == kv.end() || it->second.empty() ? def : std::atof(it->second.c_str()); };
    std::shared_ptr<Model> m;
    if (!kv["model"].empty()) {
        m = Model::load((p / upath(kv["model"])).u8string(), &err);
        if (!m) return false;
    }
    std::string data = kv["data"];
    if (!data.empty() && upath(data).is_relative()) data = (p / upath(data)).u8string();
    int sub = (int)num("subword", 0);
    A.subword = sub > 0;
    if (sub > 0) A.subword_size = sub;
    A.img_size = (int)num("img_size", A.img_size);
    A.img_colors = (int)num("img_colors", A.img_colors);
    A.img_mirror = num("img_mirror", 1) != 0;
    A.snd_rate = (int)num("snd_rate", A.snd_rate);
    A.model = m;
    if (m) A.spec = m->spec;
    A.from_file = (bool)m;
    A.ft_added = A.ft_dropped = 0;
    reset_generation();
    if (!data.empty() && fs::exists(upath(data), ec)) {
        int kind = std::max(0, std::min(3, (int)num("kind", 0)));
        A.data_kind = kind;
        std::snprintf(A.data_path, sizeof A.data_path, "%s", data.c_str());
        if (!load_data_impl(data, kind, (int)num("target", -1))) err = A.data_msg;
        set_tests(tests);
    } else {
        free_textures(A.data_tex);
        A.data = Dataset();
        A.data_path[0] = 0;
        A.q_tag.clear();
        A.a_tag.clear();
        set_tests(tests);
        if (!data.empty()) err = strf(tr(S::ResumeNoData), data.c_str());
    }
    A.stop_on_pass = num("stop_on_pass", 0) != 0;
    if (kv.count("cfg.steps")) {
        TrainConfig& c = A.cfg;
        c.steps = std::max(1, (int)num("cfg.steps", c.steps));
        c.batch = std::max(1, (int)num("cfg.batch", c.batch));
        c.accum = std::max(1, (int)num("cfg.accum", c.accum));
        c.lr = (float)num("cfg.lr", c.lr);
        c.weight_decay = (float)num("cfg.wd", c.weight_decay);
        c.eval_every = std::max(1, (int)num("cfg.eval_every", c.eval_every));
        c.patience = (int)num("cfg.patience", c.patience);
        c.optimizer = (int)num("cfg.optimizer", c.optimizer);
        c.dropout = (float)num("cfg.dropout", c.dropout);
        c.prompt_weight = (float)num("cfg.prompt_weight", c.prompt_weight);
        c.align = (float)num("cfg.align", c.align);
        c.ema = (float)num("cfg.ema", c.ema);
        c.keep_best = num("cfg.keep_best", 1) != 0;
        c.probe_every = (int)num("cfg.probe_every", c.probe_every);
    }
    A.project_dir = dir;
    A.project_name = kv["name"].empty() ? p.filename().u8string() : kv["name"];
    A.project_desc = kv["desc"];
    remember_project(dir);
    A.projects_scanned = false;
    add_activity(State::ACT_PROJECT_OPENED, A.project_name);
    return true;
}

void scan_projects() {
    namespace fs = std::filesystem;
    A.projects.clear();
    A.projects_scanned = true;
    std::set<std::string> seen;
    auto add = [&](const fs::path& dir) {
        std::error_code ec;
        std::string key = fs::weakly_canonical(dir, ec).u8string();
        if (!seen.insert(key).second) return;
        std::map<std::string, std::string> kv;
        std::string tests;
        if (!read_project(dir.u8string(), kv, tests)) return;
        State::ProjectInfo pi;
        pi.dir = dir.u8string();
        pi.name = kv["name"].empty() ? dir.filename().u8string() : kv["name"];
        pi.updated = (std::time_t)std::atoll(kv["updated"].c_str());
        pi.desc = kv["desc"];
        pi.data = file_name(kv["data"]);
        pi.data_path = kv["data"].empty() || !upath(kv["data"]).is_relative() ? kv["data"] : (dir / upath(kv["data"])).u8string();
        pi.kind = kv["data"].empty() ? -1 : std::atoi(kv["kind"].c_str());
        pi.model = kv["model"];
        pi.params = std::atof(kv["params"].c_str());
        A.projects.push_back(pi);
    };
    std::error_code ec;
    for (fs::directory_iterator it(upath(projects_root()), ec), end; !ec && it != end; it.increment(ec))
        if (it->is_directory(ec)) add(it->path());
    for (const std::string& r : A.recent_projects) add(upath(r));
    std::sort(A.projects.begin(), A.projects.end(),
              [](const State::ProjectInfo& a, const State::ProjectInfo& b) { return a.updated > b.updated; });
}

void ask_open_project() {
    std::string p = open_file_dialog(DLG_PROJECT);
    if (p.empty()) return;
    std::string err;
    A.project_ok = open_project_dir(p, err) && err.empty();
    A.project_msg = err;
    if (A.project_ok || !A.project_dir.empty()) A.page = PAGE_PROJECTS;
}

void ask_new_project() {
    std::string base = A.data.kind != DATA_NONE ? upath(A.data.path).stem().u8string() : std::string(tr(S::NewProject));
    std::snprintf(A.new_project_name, sizeof A.new_project_name, "%s", base.c_str());
    A.new_project_desc[0] = 0;
    A.new_project_err.clear();
    A.want_new_project = true;
}

void open_terminal_here() { open_terminal(A.project_dir.empty() ? exe_dir() : A.project_dir); }

void save_or_create_project() {
    if (A.project_dir.empty()) ask_new_project();
    else save_open_project();
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

// Offers to continue a run that was cut off (power cut, crash, closed app).
void resume_card() {
    if (!A.resume_avail || busy()) return;
    begin_card("resume", tr(S::ResumeTitle));
    text_wrapped(strf(tr(S::ResumeText), A.resume_info.step, A.resume_info.cfg.steps, fmt_time(A.resume_info.elapsed).c_str()).c_str(), C::text);
    if (!A.resume_data.empty()) text_wrapped(strf(tr(S::ResumeData), A.resume_data.c_str()).c_str(), C::muted, F.small);
    if (A.resume_warn) text_wrapped(tr(S::ResumeWarn), C::warn, F.small);
    if (!A.resume_msg.empty()) text_wrapped(strf(tr(S::ResumeFailed), A.resume_msg.c_str()).c_str(), C::bad, F.small);
    gap(4);
    if (button(tr(S::ResumeGo), BTN_PRIMARY, IC_PLAY)) resume_training();
    ImGui::SameLine();
    if (button(tr(S::ResumeDiscard), BTN_GHOST, IC_TRASH)) discard_resume();
    end_card();
    gap();
}

// ------------------------------------------------------------------- home

// Polled while the home page is shown: machine load every second, saved
// models every few seconds.
void refresh_home_numbers() {
    double now = ImGui::GetTime();
    if (now - A.usage_at >= 1.0) {
        Usage u = sample_usage();
        if (u.cpu < 0) u.cpu = A.usage.cpu;
        A.usage = u;
        A.usage_at = now;
    }
    if (!A.projects_scanned) {
        scan_projects();
        A.models_counted_at = -100;
    }
    if (now - A.models_counted_at >= 5.0) {
        namespace fs = std::filesystem;
        A.models_counted_at = now;
        A.model_count = 0;
        std::error_code ec;
        for (fs::directory_iterator it(upath(models_dir()), ec), end; !ec && it != end; it.increment(ec))
            if (it->path().extension() == ".kor") ++A.model_count;
        for (auto& pi : A.projects)
            if (!pi.model.empty()) ++A.model_count;
        // every training run writes one log file
        A.train_runs = 0;
        for (fs::directory_iterator it(upath(logs_dir()), ec), end; !ec && it != end; it.increment(ec))
            if (it->path().extension() == ".log") ++A.train_runs;
        // datasets: loaded before or kept in a project
        std::set<std::string> seen;
        A.dataset_count = 0;
        A.data_bytes = 0;
        auto add = [&](const std::string& p) {
            if (p.empty()) return;
            std::error_code e;
            std::string key = fs::weakly_canonical(upath(p), e).u8string();
            if (!fs::exists(upath(p), e) || !seen.insert(key).second) return;
            ++A.dataset_count;
            if (fs::is_regular_file(upath(p), e)) {
                A.data_bytes += (double)fs::file_size(upath(p), e);
                return;
            }
            int n = 0;
            for (fs::recursive_directory_iterator it(upath(p), e), end; !e && it != end && n < 20000; it.increment(e), ++n)
                if (it->is_regular_file(e)) A.data_bytes += (double)it->file_size(e);
        };
        for (auto& p : A.recent_data) add(p);
        for (auto& pi : A.projects) add(pi.data_path);
    }
}

void refresh_disk() {
    double now = ImGui::GetTime();
    if (now - A.disk_at < 10.0) return;
    A.disk_at = now;
    std::error_code ec;
    auto sp = std::filesystem::space(upath(documents_dir()), ec);
    if (!ec) {
        A.disk_total_gb = sp.capacity / 1e9;
        A.disk_used_gb = (sp.capacity - sp.available) / 1e9;
    }
}

// "72%", or "%72" in Turkish
std::string pct(double frac) {
    int v = (int)std::lround(std::max(0.0, std::min(1.0, frac)) * 100);
    return get_language() == LANG_TR ? strf("%%%d", v) : strf("%d%%", v);
}

std::string fmt_bytes(double b) {
    if (b >= 1e9) return strf("%.1f GB", b / 1e9);
    if (b >= 1e6) return strf("%.1f MB", b / 1e6);
    return strf("%.0f KB", b / 1e3);
}

// "NVIDIA GeForce RTX 3050 Ti Laptop GPU" -> "RTX 3050 Ti"
std::string short_gpu(std::string s) {
    for (const char* cut : {"(R)", "(TM)", "NVIDIA ", "GeForce ", "AMD ", "Intel ", " Laptop GPU", " GPU"}) {
        size_t at;
        while ((at = s.find(cut)) != std::string::npos) s.erase(at, std::strlen(cut));
    }
    while (!s.empty() && s.front() == ' ') s.erase(s.begin());
    return s;
}

void load_hero() {
    Image img;
    std::string err;
    for (const std::string& p : {(upath(exe_dir()) / "assets" / "hero.png").u8string(), std::string("assets/hero.png")})
        if (load_image(p, img, &err)) {
            A.hero_tex = make_texture(img, true);
            A.hero_w = img.w;
            A.hero_h = img.h;
            return;
        }
}

// Moon over mountains and a lake, drawn with shapes (no image files).
void hero_scene(ImDrawList* dl, ImVec2 a, ImVec2 b, float r) {
    const float pi = 3.14159265f;
    float w = b.x - a.x, h = b.y - a.y, tm = (float)ImGui::GetTime();
    float hz = a.y + h * 0.70f;  // horizon
    // vertical gradient over the vertices added since v0
    auto shade = [&](int v0, float y0, float y1, const ImVec4& c0, const ImVec4& c1) {
        for (int i = v0; i < dl->VtxBuffer.Size; ++i) {
            ImDrawVert& v = dl->VtxBuffer[i];
            float t = std::max(0.f, std::min(1.f, (v.pos.y - y0) / std::max(1.f, y1 - y0)));
            v.col = col(mix(c0, c1, t), ((v.col >> IM_COL32_A_SHIFT) & 0xFF) / 255.f);
        }
    };
    int v0 = dl->VtxBuffer.Size;
    dl->AddRectFilled(a, b, IM_COL32_WHITE, r);
    shade(v0, a.y, hz, ImVec4(0.06f, 0.04f, 0.14f, 1), ImVec4(0.33f, 0.14f, 0.45f, 1));

    unsigned seed = 2024;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (seed >> 8) / 16777216.f; };
    for (int i = 0; i < 80; ++i) {
        float x = a.x + w * (0.30f + 0.70f * rnd()), y = a.y + (hz - a.y) * 0.8f * rnd();
        float sz = px(0.5f + 1.1f * rnd()), ph = rnd() * 6.28f;
        float al = (0.35f + 0.45f * (0.5f + 0.5f * std::sin(tm * 1.2f + ph))) * ((x - a.x) / w);
        dl->AddCircleFilled(ImVec2(x, y), sz, col(ImVec4(1.f, 0.95f, 1.f, 1), al), 8);
    }

    ImVec2 mc(a.x + w * 0.72f, a.y + h * 0.30f);
    float mr = h * 0.11f;
    for (int i = 7; i >= 1; --i) dl->AddCircleFilled(mc, mr * (1.f + i * 0.5f), col(ImVec4(0.80f, 0.50f, 1.f, 1), 0.035f), 48);
    dl->AddCircleFilled(mc, mr, col(ImVec4(1.f, 0.95f, 0.99f, 1)), 48);
    dl->AddCircleFilled(ImVec2(mc.x + mr * 0.3f, mc.y - mr * 0.2f), mr * 0.16f, col(ImVec4(0.92f, 0.86f, 0.96f, 1)), 16);
    dl->AddCircleFilled(ImVec2(mc.x - mr * 0.35f, mc.y + mr * 0.3f), mr * 0.1f, col(ImVec4(0.92f, 0.86f, 0.96f, 1)), 12);

    // pts: x (0..1) and height (0..1) pairs; dir -1 draws the reflection in the lake
    auto ridge = [&](const float* pts, int n, float amp, const ImVec4& c0, const ImVec4& c1, float dir, float alpha) {
        dl->PathClear();
        dl->PathLineTo(ImVec2(a.x, hz));
        for (int i = 0; i < n; ++i) dl->PathLineTo(ImVec2(a.x + w * pts[2 * i], hz - dir * amp * pts[2 * i + 1]));
        dl->PathLineTo(ImVec2(b.x, hz));
        int v = dl->VtxBuffer.Size;
        dl->PathFillConcave(col(c0, alpha));
        if (dir > 0) shade(v, hz - amp, hz, c0, c1);
    };
    static const float far_[] = {0, .35f, .08f, .52f, .17f, .38f, .27f, .72f, .36f, .45f, .46f, .64f, .55f, .40f,
                                 .63f, .80f, .72f, .52f, .80f, .68f, .88f, .42f, .95f, .58f, 1, .45f};
    static const float near_[] = {0, .30f, .10f, .16f, .20f, .44f, .30f, .20f, .42f, .34f, .50f, .12f,
                                  .60f, .38f, .70f, .18f, .80f, .42f, .90f, .22f, 1, .36f};
    float far_amp = h * 0.32f, near_amp = h * 0.20f;
    ridge(far_, 13, far_amp, ImVec4(0.36f, 0.20f, 0.56f, 1), ImVec4(0.22f, 0.12f, 0.38f, 1), 1, 1);
    ridge(near_, 11, near_amp, ImVec4(0.19f, 0.10f, 0.33f, 1), ImVec4(0.12f, 0.07f, 0.22f, 1), 1, 1);

    v0 = dl->VtxBuffer.Size;
    dl->AddRectFilled(ImVec2(a.x, hz), b, IM_COL32_WHITE);
    shade(v0, hz, b.y, ImVec4(0.22f, 0.11f, 0.36f, 1), ImVec4(0.05f, 0.03f, 0.10f, 1));
    ridge(near_, 11, near_amp * 0.6f, ImVec4(0.30f, 0.16f, 0.48f, 1), ImVec4(), -1, 0.35f);
    for (int i = 0; i < 9; ++i) {
        float y = hz + px(5) + i * (b.y - hz) / 10.f;
        float hw = mr * (0.9f - i * 0.07f) * (0.75f + 0.25f * std::sin(tm * 1.5f + i * 1.3f));
        dl->AddLine(ImVec2(mc.x - hw, y), ImVec2(mc.x + hw, y), col(ImVec4(1.f, 0.88f, 1.f, 1), 0.38f - i * 0.035f), px(1.5f));
    }

    // dark on the left so the title reads well
    ImU32 dark = col(ImVec4(0.04f, 0.03f, 0.09f, 1), 0.92f), clear = col(ImVec4(0.04f, 0.03f, 0.09f, 1), 0.f);
    dl->AddRectFilledMultiColor(a, ImVec2(a.x + w * 0.62f, b.y), dark, clear, clear, dark);

    // round the corners again by painting the page colour outside them
    ImU32 page = col(C::bg);
    auto corner = [&](ImVec2 c, ImVec2 center, float a0) {
        dl->PathClear();
        dl->PathLineTo(c);
        dl->PathArcTo(center, r, a0, a0 + pi * 0.5f, 10);
        dl->PathFillConcave(page);
    };
    corner(a, ImVec2(a.x + r, a.y + r), pi);
    corner(ImVec2(b.x, a.y), ImVec2(b.x - r, a.y + r), pi * 1.5f);
    corner(b, ImVec2(b.x - r, b.y - r), 0);
    corner(ImVec2(a.x, b.y), ImVec2(a.x + r, b.y - r), pi * 0.5f);
    dl->AddRect(a, b, col(C::border_hi, 0.7f), r);
}

void hero(float w) {
    ImFont* hf = F.hero ? F.hero : ImGui::GetFont();
    ImFont* body = ImGui::GetFont();
    float pad = px(28);
    float tw = w < px(620) ? w - pad * 2 : std::min(w * 0.6f, px(480));
    const char* sub = tr(S::HeroSub);
    float sub_h = text_size(body, sub, nullptr, tw).y;
    const char* labels[3] = {tr(S::NewModel), tr(S::OpenProject), tr(S::ChatWithModel)};
    static const Icon ics[3] = {IC_PLAY, IC_FOLDER, IC_CHAT};
    float bh = px(38), bg = px(10), bx[3], by[3], x = 0, y = 0;
    for (int i = 0; i < 3; ++i) {
        float bw = button_width(labels[i], ics[i]);
        if (x > 0 && x + bw > w - pad * 2) {
            x = 0;
            y += bh + bg;
        }
        bx[i] = x;
        by[i] = y;
        x += bw + bg;
    }
    float content = hf->FontSize + px(8) + sub_h + px(22) + y + bh;
    float h = std::max(px(216), content + pad * 2);
    ImVec2 a = ImGui::GetCursorScreenPos(), b(a.x + w, a.y + h);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float r = px(14);
    if (A.hero_tex) {
        // fill the box (crop the picture, keep the moon side)
        ImVec2 uv0(0, 0), uv1(1, 1);
        float box = w / h, pic = (float)A.hero_w / std::max(1, A.hero_h);
        if (pic > box) {
            float f = box / pic;
            uv0.x = (1 - f) * 0.7f;
            uv1.x = uv0.x + f;
        } else {
            float f = pic / box;
            uv0.y = (1 - f) * 0.5f;
            uv1.y = uv0.y + f;
        }
        dl->AddImageRounded(A.hero_tex, a, b, uv0, uv1, IM_COL32_WHITE, r);
        // darker on the left so the text reads well
        int v0 = dl->VtxBuffer.Size;
        dl->AddRectFilled(a, b, IM_COL32_WHITE, r);
        for (int i = v0; i < dl->VtxBuffer.Size; ++i) {
            ImDrawVert& v = dl->VtxBuffer[i];
            float t = std::max(0.f, 1.f - (v.pos.x - a.x) / (w * 0.7f));
            float al = ((v.col >> IM_COL32_A_SHIFT) & 0xFF) / 255.f;
            v.col = col(ImVec4(0.04f, 0.02f, 0.09f, 1), al * (0.15f + 0.6f * t));
        }
        dl->AddRect(a, b, col(C::border_hi, 0.6f), r);
    } else {
        hero_scene(dl, a, b, r);
    }
    float ty = a.y + (h - content) * 0.5f;
    dl->AddText(hf, hf->FontSize, ImVec2(a.x + pad, ty), col(C::text), "KOR ");
    dl->AddText(hf, hf->FontSize, ImVec2(a.x + pad + text_size(hf, "KOR ").x, ty), col(C::accent_hi), "Studio");
    ty += hf->FontSize + px(8);
    dl->AddText(body, body->FontSize, ImVec2(a.x + pad, ty), col(ImVec4(0.90f, 0.87f, 0.97f, 1)), sub, nullptr, tw);
    ty += sub_h + px(22);
    for (int i = 0; i < 3; ++i) {
        ImGui::SetCursorScreenPos(ImVec2(a.x + pad + bx[i], ty + by[i]));
        ImGui::PushID(i);
        if (button(labels[i], i == 0 ? BTN_PRIMARY : BTN_SECONDARY, ics[i], 0, bh)) {
            if (i == 0) A.page = PAGE_DATA;
            if (i == 1) ask_open_project();
            if (i == 2) A.page = PAGE_CHAT;
        }
        ImGui::PopID();
    }
    ImGui::SetCursorScreenPos(a);
    ImGui::Dummy(ImVec2(w, h));
}

// A clickable row: tinted icon, label (and a second line), then a text or a chevron on the right.
bool list_row(const char* id, Icon ic, const ImVec4& tint, const char* label, const char* sub = nullptr,
              const char* right = nullptr, const ImVec4& right_col = C::subtle) {
    float w = ImGui::GetContentRegionAvail().x;
    ImFont* bf = ImGui::GetFont();
    ImFont* sf = F.small ? F.small : bf;
    float h = sub && *sub ? px(50) : px(40);
    ImVec2 p = ImGui::GetCursorScreenPos();
    bool clicked = ImGui::InvisibleButton(id, ImVec2(w, h));
    bool hov = ImGui::IsItemHovered();
    if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    float t = anim(ImGui::GetItemID(), hov);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (t > 0.01f) dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(C::surface2, t), px(8));
    float s = px(30);
    ImVec2 ib(p.x + px(6), p.y + (h - s) * 0.5f);
    dl->AddRectFilled(ib, ImVec2(ib.x + s, ib.y + s), col(tint, 0.15f), px(8));
    icon(dl, ic, ImVec2(ib.x + s * 0.5f, ib.y + s * 0.5f), px(16), col(tint));
    float x = ib.x + s + px(12);
    float rw = right ? text_size(sf, right).x + px(16) : px(26);
    float maxw = p.x + w - x - rw;
    if (sub && *sub) {
        float y0 = p.y + (h - bf->FontSize - sf->FontSize - px(3)) * 0.5f;
        text_ellipsis(dl, bf, ImVec2(x, y0), maxw, label, col(C::text));
        text_ellipsis(dl, sf, ImVec2(x, y0 + bf->FontSize + px(3)), maxw, sub, col(C::muted));
    } else {
        text_ellipsis(dl, bf, ImVec2(x, p.y + (h - bf->FontSize) * 0.5f), maxw, label, col(C::text));
    }
    if (right)
        dl->AddText(sf, sf->FontSize, ImVec2(p.x + w - px(10) - text_size(sf, right).x, p.y + (h - sf->FontSize) * 0.5f), col(right_col), right);
    else
        icon(dl, IC_CHEVRON_RIGHT, ImVec2(p.x + w - px(14), p.y + h * 0.5f), px(14), col(mix(C::subtle, C::text, t)));
    return clicked;
}

// Title line of a card opened with begin_card(id): an icon and the title.
void card_header(Icon ic, const char* title) {
    ImFont* bf = F.bold ? F.bold : ImGui::GetFont();
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    icon(dl, ic, ImVec2(p.x + px(8), p.y + bf->FontSize * 0.5f), px(16), col(C::accent_hi));
    dl->AddText(bf, bf->FontSize, ImVec2(p.x + px(26), p.y), col(C::text), title);
    ImGui::Dummy(ImVec2(0, bf->FontSize));
    gap(6);
}

// A small link with an arrow on the right of the current card's title line.
bool card_link(const char* label) {
    ImFont* f = F.small ? F.small : ImGui::GetFont();
    ImFont* tf = F.bold ? F.bold : ImGui::GetFont();
    ImVec2 ts = text_size(f, label), back = ImGui::GetCursorScreenPos(), wp = ImGui::GetWindowPos();
    float w = ts.x + px(18);
    ImVec2 p(wp.x + ImGui::GetWindowWidth() - card_padding_x() - w, wp.y + card_padding_y() + (tf->FontSize - ts.y) * 0.5f);
    ImGui::SetCursorScreenPos(p);
    bool clicked = ImGui::InvisibleButton("##link", ImVec2(w, ts.y));
    bool hov = ImGui::IsItemHovered();
    if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImU32 c = col(hov ? C::text : mix(C::muted, C::text, 0.4f));
    dl->AddText(f, f->FontSize, p, c, label);
    icon(dl, IC_ARROW_RIGHT, ImVec2(p.x + ts.x + px(11), p.y + ts.y * 0.5f), px(12), c);
    ImGui::SetCursorScreenPos(back);
    return clicked;
}

Icon project_icon(int kind) {
    switch (kind) {
        case 0: return IC_CHATS;
        case 1: return IC_TABLE;
        case 2: return IC_LAYERS;
        case 3: return IC_WAVE;
    }
    return IC_CUBE;
}

std::string project_desc(const State::ProjectInfo& pi) {
    if (!pi.desc.empty()) return pi.desc;
    if (pi.params > 0) return strf(tr(S::ProjectParamsDesc), fmt_count(pi.params).c_str());
    return pi.data.empty() ? tr(S::ProjectNoModel) : pi.data;
}

// Rounded square with an icon, as in the project lists.
void icon_box(ImDrawList* dl, ImVec2 p, float s, Icon ic, bool hi) {
    dl->AddRectFilled(p, ImVec2(p.x + s, p.y + s), col(hi ? mix(C::surface3, C::accent, 0.25f) : C::surface2), px(8));
    dl->AddRect(p, ImVec2(p.x + s, p.y + s), col(C::border_hi), px(8));
    icon(dl, ic, ImVec2(p.x + s * 0.5f, p.y + s * 0.5f), s * 0.5f, col(C::text));
}

void project_open_from_list(const std::string& dir) {
    if (dir == A.project_dir) {
        A.page = PAGE_PROJECTS;
        return;
    }
    std::string err;
    A.project_ok = open_project_dir(dir, err) && err.empty();
    A.project_msg = err;
    A.page = PAGE_PROJECTS;
}

const ImVec4& project_tint(size_t i) {
    static const ImVec4* tints[] = {&C::accent, &C::blue, &C::teal, &C::violet, &C::warn};
    return *tints[i % 5];
}

void recent_projects_card(float min_h, float width, float* natural) {
    begin_card("recentp", nullptr, min_h, width);
    card_header(IC_FOLDER, tr(S::RecentProjects));
    if (card_link(tr(S::SeeAll))) A.page = PAGE_PROJECTS;
    if (A.projects.empty()) {
        text_wrapped(tr(S::NoProjects), C::muted, F.small);
        gap(6);
        if (button(tr(S::NewProject), BTN_SECONDARY, IC_PLUS)) ask_new_project();
        end_card(natural);
        return;
    }
    ImFont* bf = F.bold ? F.bold : ImGui::GetFont();
    ImFont* sf = F.small ? F.small : ImGui::GetFont();
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(px(8), 0));
    size_t n = std::min<size_t>(A.projects.size(), 4);
    for (size_t i = 0; i < n; ++i) {
        const State::ProjectInfo& pi = A.projects[i];
        ImGui::PushID((int)i);
        float w = ImGui::GetContentRegionAvail().x, h = px(52);
        ImVec2 p = ImGui::GetCursorScreenPos();
        bool click = ImGui::InvisibleButton("##p", ImVec2(w, h));
        bool hov = ImGui::IsItemHovered(), cur = pi.dir == A.project_dir;
        if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        float t = anim(ImGui::GetItemID(), hov);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        if (t > 0.01f) dl->AddRectFilled(ImVec2(p.x - px(8), p.y), ImVec2(p.x + w + px(8), p.y + h), col(C::surface2, t), px(8));
        if (i + 1 < n) dl->AddLine(ImVec2(p.x, p.y + h - 0.5f), ImVec2(p.x + w, p.y + h - 0.5f), col(C::border));
        float s = px(34);
        icon_box(dl, ImVec2(p.x, p.y + (h - s) * 0.5f), s, project_icon(pi.kind), cur);
        std::string when = ago(pi.updated);
        float ww = text_size(sf, when.c_str()).x;
        float x = p.x + s + px(14), tw = p.x + w - x - ww - px(12);
        float y0 = p.y + (h - bf->FontSize - sf->FontSize - px(3)) * 0.5f;
        text_ellipsis(dl, bf, ImVec2(x, y0), tw, pi.name.c_str(), col(C::text));
        text_ellipsis(dl, sf, ImVec2(x, y0 + bf->FontSize + px(3)), tw, project_desc(pi).c_str(), col(C::muted));
        dl->AddText(sf, sf->FontSize, ImVec2(p.x + w - ww, p.y + (h - sf->FontSize) * 0.5f), col(cur ? C::good : C::muted), when.c_str());
        if (click) project_open_from_list(pi.dir);
        ImGui::PopID();
    }
    ImGui::PopStyleVar();
    end_card(natural);
}

// The full-width outlined button at the bottom of a card.
bool outline_button(const char* label, Icon trailing) {
    ImFont* f = ImGui::GetFont();
    float w = ImGui::GetContentRegionAvail().x, h = px(34);
    ImVec2 p = ImGui::GetCursorScreenPos();
    bool click = ImGui::InvisibleButton(label, ImVec2(w, h));
    bool hov = ImGui::IsItemHovered();
    if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    float t = anim(ImGui::GetItemID(), hov);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(mix(C::surface2, C::surface3, t)), px(8));
    dl->AddRect(p, ImVec2(p.x + w, p.y + h), col(C::border_hi), px(8));
    ImVec2 ts = text_size(f, label);
    float iw = trailing != IC_NONE ? px(20) : 0, x = p.x + (w - ts.x - iw) * 0.5f;
    dl->AddText(f, f->FontSize, ImVec2(x, p.y + (h - ts.y) * 0.5f), col(mix(C::muted, C::text, 0.5f + 0.5f * t)), label);
    if (trailing != IC_NONE) icon(dl, trailing, ImVec2(x + ts.x + px(12), p.y + h * 0.5f), px(12), col(C::muted));
    return click;
}

void activity_card(float min_h, float width, float* natural) {
    begin_card("activity", nullptr, min_h, width);
    card_header(IC_WAVE, tr(S::Activity));
    ImFont* bf = ImGui::GetFont();
    ImFont* sf = F.small ? F.small : bf;
    float rh = px(34);
    auto row = [&](const ImVec4& dot, const std::string& label, std::function<void(ImDrawList*, ImVec2, float)> right, float right_w, int page) {
        float w = ImGui::GetContentRegionAvail().x;
        ImVec2 p = ImGui::GetCursorScreenPos();
        bool click = ImGui::InvisibleButton(label.c_str(), ImVec2(w, rh));
        if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        float cy = p.y + rh * 0.5f;
        dl->AddCircleFilled(ImVec2(p.x + px(5), cy), px(4), col(dot));
        text_ellipsis(dl, bf, ImVec2(p.x + px(20), cy - bf->FontSize * 0.5f), w - px(28) - right_w, label.c_str(), col(C::text));
        right(dl, ImVec2(p.x + w, cy), right_w);
        if (click) A.page = page;
    };
    auto check = [&](ImDrawList* dl, ImVec2 rc, float) {
        ImVec2 c(rc.x - px(10), rc.y);
        dl->AddCircle(c, px(9), col(C::border_hi), 0, 1.2f);
        icon(dl, IC_CHECK, c, px(11), col(C::muted));
    };
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(px(8), px(2)));
    if (busy()) {
        TrainStats st = A.trainer.stats();
        float frac = (float)st.step / std::max(1, st.total);
        std::string label = strf(tr(S::ActTrainRunning), file_name(A.data.path).c_str()), pc = pct(frac);
        row(C::good, label, [&](ImDrawList* dl, ImVec2 rc, float rw) {
            float x0 = rc.x - rw, bw = rw;
            dl->AddText(sf, sf->FontSize, ImVec2(rc.x - text_size(sf, pc.c_str()).x, rc.y - sf->FontSize - px(1)), col(C::text), pc.c_str());
            dl->AddRectFilled(ImVec2(x0, rc.y + px(3)), ImVec2(x0 + bw, rc.y + px(7)), col(C::surface3), px(2));
            dl->AddRectFilled(ImVec2(x0, rc.y + px(3)), ImVec2(x0 + std::max(px(4), bw * frac), rc.y + px(7)), col(C::accent), px(2));
        }, px(100), PAGE_TRAIN);
    }
    static const int pages[State::ACT_COUNT] = {PAGE_TRAIN, PAGE_TRAIN, PAGE_TRAIN, PAGE_TRAIN, PAGE_CHAT, PAGE_DATA, PAGE_PROJECTS, PAGE_PROJECTS};
    int shown = 0, limit = busy() ? 3 : 4;
    for (auto it = A.activity.rbegin(); it != A.activity.rend() && shown < limit; ++it, ++shown) {
        int k = std::max(0, std::min((int)State::ACT_COUNT - 1, it->kind));
        ImGui::PushID(shown);
        const ImVec4& dot = k == State::ACT_TRAIN_STOP ? C::warn : k == State::ACT_TRAIN_DONE ? C::good : C::accent;
        std::string label = act_text(*it);
        if (k == State::ACT_TRAIN_STOP) {
            row(dot, label, [&](ImDrawList* dl, ImVec2 rc, float) {
                ImVec2 c(rc.x - px(10), rc.y);
                dl->AddCircle(c, px(9), col(C::border_hi), 0, 1.2f);
                icon(dl, IC_STOP, c, px(9), col(C::warn));
            }, px(22), pages[k]);
        } else {
            row(dot, label, check, px(22), pages[k]);
        }
        if (ImGui::IsItemHovered()) tooltip(ago(it->t).c_str());
        ImGui::PopID();
    }
    ImGui::PopStyleVar();
    if (A.activity.empty() && !busy()) text_wrapped(tr(S::NoActivity), C::muted, F.small);
    gap(8);
    if (outline_button(tr(S::AllLogs), IC_ARROW_RIGHT)) {
        std::error_code ec;
        std::filesystem::create_directories(upath(logs_dir()), ec);
        open_folder(logs_dir());
    }
    end_card(natural);
}

void usage_card() {
    begin_card("sysres");
    card_header(IC_WAVE, tr(S::SysResources));
    float cpu = A.usage.cpu;
    std::string center = cpu < 0 ? "-" : pct(cpu);
    ring(std::max(0.f, cpu), px(44), px(8), C::accent, center.c_str());
    ImGui::SameLine(0, px(16));
    ImGui::BeginGroup();
    ImFont* bf = ImGui::GetFont();
    auto metric = [&](Icon ic, const char* label, const std::string& value) {
        float w = ImGui::GetContentRegionAvail().x, h = px(29);
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(w, h));
        ImDrawList* dl = ImGui::GetWindowDrawList();
        float cy = p.y + h * 0.5f;
        icon(dl, ic, ImVec2(p.x + px(8), cy), px(15), col(C::muted));
        dl->AddText(bf, bf->FontSize, ImVec2(p.x + px(24), cy - bf->FontSize * 0.5f), col(C::text), label);
        float vw = text_size(bf, value.c_str()).x;
        dl->AddText(bf, bf->FontSize, ImVec2(p.x + w - vw, cy - bf->FontSize * 0.5f), col(C::text), value.c_str());
    };
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(px(8), 0));
    double ru = A.usage.ram_used_gb, rt = A.usage.ram_total_gb > 0 ? A.usage.ram_total_gb : A.sys.ram_total_gb;
    metric(IC_CPU, tr(S::CpuLoad), center);
    metric(IC_MEMORY, tr(S::RamLoad), strf("%.1f / %.1f GB", ru, rt));
    std::string gpu = A.usage.gpu_used_gb >= 0 ? strf("%.1f / %.1f GB", A.usage.gpu_used_gb, A.usage.gpu_total_gb)
                      : A.sys.gpus.empty() ? std::string("-") : short_gpu(A.sys.gpus[0]);
    metric(IC_GPU, "GPU", gpu);
    ImGui::PopStyleVar();
    ImGui::EndGroup();
    if (!A.sys.gpus.empty() && ImGui::IsItemHovered()) tooltip((A.sys.gpus[0] + "\n" + tr(S::TileCpuSub)).c_str());
    end_card();
}

// Outlined row button with an icon, as in the quick actions.
bool action_row(const char* label, Icon ic) {
    ImFont* f = ImGui::GetFont();
    float w = ImGui::GetContentRegionAvail().x, h = px(32);
    ImVec2 p = ImGui::GetCursorScreenPos();
    bool click = ImGui::InvisibleButton(label, ImVec2(w, h));
    bool hov = ImGui::IsItemHovered();
    if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    float t = anim(ImGui::GetItemID(), hov);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(mix(C::surface2, C::surface3, t)), px(8));
    dl->AddRect(p, ImVec2(p.x + w, p.y + h), col(mix(C::border, C::accent, t * 0.5f)), px(8));
    icon(dl, ic, ImVec2(p.x + px(20), p.y + h * 0.5f), px(15), col(C::text));
    text_ellipsis(dl, f, ImVec2(p.x + px(40), p.y + (h - f->FontSize) * 0.5f), w - px(48), label, col(C::text));
    return click;
}

void quick_actions_card() {
    begin_card("quick");
    card_header(IC_BOLT, tr(S::QuickActions));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(px(8), px(6)));
    if (action_row(tr(S::QaTrain), IC_PLAY_O)) {
        if (A.data.kind == DATA_NONE) A.page = PAGE_DATA;
        else {
            A.page = PAGE_TRAIN;
            if (!busy()) begin_training();
        }
    }
    if (action_row(tr(S::QaImport), IC_UPLOAD)) A.page = PAGE_DATA;
    if (action_row(tr(S::QaTest), IC_FLASK)) A.page = PAGE_TEST;
    if (action_row(tr(S::ChatWithModel), IC_CHAT)) A.page = PAGE_CHAT;
    if (action_row(tr(S::QaNewProject), IC_FOLDER)) ask_new_project();
    ImGui::PopStyleVar();
    end_card();
}

// A label and its key combination ("Ctrl K") in a key cap.
void shortcut_row(const char* keys, const char* label) {
    ImFont* kf = F.small_bold ? F.small_bold : ImGui::GetFont();
    ImFont* bf = ImGui::GetFont();
    float w = ImGui::GetContentRegionAvail().x, h = px(28);
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(w, h));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddText(bf, bf->FontSize, ImVec2(p.x, p.y + (h - bf->FontSize) * 0.5f), col(mix(C::muted, C::text, 0.5f)), label);
    float ch = kf->FontSize + px(10), cw = text_size(kf, keys).x + px(18);
    ImVec2 c0(p.x + w - cw, p.y + (h - ch) * 0.5f);
    dl->AddRectFilled(c0, ImVec2(c0.x + cw, c0.y + ch), col(C::surface2), px(6));
    dl->AddRect(c0, ImVec2(c0.x + cw, c0.y + ch), col(C::border_hi), px(6));
    dl->AddText(kf, kf->FontSize, ImVec2(c0.x + px(9), c0.y + px(5)), col(C::text), keys);
}

void shortcuts_card() {
    begin_card("keys");
    card_header(IC_KEYBOARD, tr(S::Shortcuts));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(px(8), px(2)));
    shortcut_row("Ctrl  K", tr(S::ScSearch));
    shortcut_row("Ctrl  `", tr(S::Terminal));
    shortcut_row("Ctrl  ,", tr(S::Settings));
    ImGui::PopStyleVar();
    end_card();
}

// One of the six tiles under the picture: icon, label, value, then a status
// dot or a usage bar.
struct Tile {
    Icon ic;
    std::string label, value, sub;
    ImVec4 dot = C::good;
    float bar = -1;  // 0..1 shows a bar with the percentage instead of the dot
    bool show_dot = true;
};

void tile(const Tile& t, float w, float h) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(w, h));
    bool hov = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(C::surface), px(12));
    dl->AddRect(p, ImVec2(p.x + w, p.y + h), col(C::border), px(12));
    ImFont* sf = F.small ? F.small : ImGui::GetFont();
    ImFont* vf = F.h2 ? F.h2 : ImGui::GetFont();
    float pad = px(12), x = p.x + pad, iw = w - pad * 2, y = p.y + pad;
    icon(dl, t.ic, ImVec2(x + px(10), y + px(10)), px(20), col(C::accent_hi));
    y += px(20) + px(8);
    text_ellipsis(dl, ImGui::GetFont(), ImVec2(x, y), iw, t.label.c_str(), col(mix(C::muted, C::text, 0.6f)));
    y += ImGui::GetFontSize() + px(4);
    bool cut = text_ellipsis(dl, vf, ImVec2(x, y), iw, t.value.c_str(), col(C::text));
    y += vf->FontSize + px(8);
    if (t.bar >= 0) {
        std::string pc = pct(t.bar);
        float pw = text_size(sf, pc.c_str()).x, bw = iw - pw - px(8), by = y + sf->FontSize * 0.5f;
        dl->AddRectFilled(ImVec2(x, by - px(2)), ImVec2(x + bw, by + px(2)), col(C::surface3), px(2));
        dl->AddRectFilled(ImVec2(x, by - px(2)), ImVec2(x + std::max(px(4), bw * t.bar), by + px(2)), col(C::accent), px(2));
        dl->AddText(sf, sf->FontSize, ImVec2(x + iw - pw, y), col(C::muted), pc.c_str());
    } else {
        float tx = x;
        if (t.show_dot) {
            dl->AddCircleFilled(ImVec2(x + px(3), y + sf->FontSize * 0.5f), px(3), col(t.dot));
            tx += px(12);
        }
        text_ellipsis(dl, sf, ImVec2(tx, y), x + iw - tx, t.sub.c_str(), col(t.show_dot ? t.dot : C::muted));
    }
    if (hov && cut) tooltip(t.value.c_str());
}

void home_tiles(float w) {
    std::vector<Tile> ts;
    Tile m{IC_CUBE, tr(S::TileModels), std::to_string(A.model_count), A.model ? tr(S::TileActive) : tr(S::TileNoModel)};
    m.dot = A.model ? C::good : C::subtle;
    ts.push_back(m);
    Tile d{IC_LAYERS, tr(S::TileDatasets), std::to_string(A.dataset_count), strf(tr(S::TileTotal), fmt_bytes(A.data_bytes).c_str())};
    d.show_dot = false;
    ts.push_back(d);
    Tile r{IC_PLAY_O, tr(S::TileTrainings), std::to_string(A.train_runs), busy() ? tr(S::TileRunning) : tr(S::TileReady)};
    r.dot = busy() ? C::violet : C::good;
    ts.push_back(r);
    Tile c{IC_CPU, "vCPU", std::to_string(A.sys.logical), ""};
    c.bar = std::max(0.f, A.usage.cpu);
    ts.push_back(c);
    double rt = A.usage.ram_total_gb > 0 ? A.usage.ram_total_gb : A.sys.ram_total_gb;
    Tile m2{IC_MEMORY, "RAM", strf("%.1f GB", rt), ""};
    m2.bar = rt > 0 ? (float)(A.usage.ram_used_gb / rt) : 0.f;
    ts.push_back(m2);
    Tile g{IC_GPU, "GPU", A.sys.gpus.empty() ? std::string(tr(S::TileNoGpu)) : short_gpu(A.sys.gpus[0]),
           A.sys.gpus.empty() ? "-" : tr(S::TileDetected)};
    g.dot = A.sys.gpus.empty() ? C::subtle : C::good;
    ts.push_back(g);

    float gap_ = px(10);
    int cols = std::clamp((int)((w + gap_) / (px(96) + gap_)), 1, 6);
    if (cols == 4 || cols == 5) cols = 3;
    float cw = std::floor((w - gap_ * (cols - 1)) / cols);
    ImFont* vf = F.h2 ? F.h2 : ImGui::GetFont();
    float h = px(12) * 2 + px(20) + px(8) + ImGui::GetFontSize() + px(4) + vf->FontSize + px(8) + (F.small ? F.small->FontSize : 13.f);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(gap_, gap_));
    for (size_t i = 0; i < ts.size(); ++i) {
        if (i % cols) ImGui::SameLine(0, gap_);
        ImGui::PushID((int)i);
        tile(ts[i], cw, h);
        ImGui::PopID();
    }
    ImGui::PopStyleVar();
}

void home_main(float w) {
    hero(w);
    if (A.resume_avail && !busy()) resume_card();
    home_tiles(w);
    gap(2);

    static EqualRow row;
    float g = px(12);
    bool two = w > px(640);
    float cw = two ? std::floor((w - g) * 0.5f) : 0;
    recent_projects_card(two ? row.prev : 0, cw, row.track());
    if (two) ImGui::SameLine(0, g);
    else gap();
    activity_card(two ? row.prev : 0, cw, row.track());
    row.next();
}

void home_side() {
    usage_card();
    quick_actions_card();
    shortcuts_card();
}

void page_home() {
    refresh_home_numbers();
    CardPadding cp(16, 14);
    float w = ImGui::GetContentRegionAvail().x, side = px(272), g = px(12);
    if (w < px(980)) {
        home_main(w);
        gap();
        home_side();
        return;
    }
    ImGuiWindowFlags cf = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
    ImGui::BeginChild("##main", ImVec2(w - side - g, 0), ImGuiChildFlags_AutoResizeY, cf);
    home_main(w - side - g);
    ImGui::EndChild();
    ImGui::SameLine(0, g);
    ImGui::BeginChild("##side", ImVec2(side, 0), ImGuiChildFlags_AutoResizeY, cf);
    home_side();
    ImGui::EndChild();
}

// --------------------------------------------------------------- projects

void project_tile(size_t i, float w, float h) {
    const State::ProjectInfo& pi = A.projects[i];
    bool cur = pi.dir == A.project_dir;
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::SetNextItemAllowOverlap();
    bool open = ImGui::InvisibleButton("##tile", ImVec2(w, h));
    bool hov = ImGui::IsItemHovered();
    float t = anim(ImGui::GetItemID(), hov || cur);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const ImVec4& tint = project_tint(i);
    dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(mix(C::surface, C::surface2, t)), px(12));
    dl->AddRect(p, ImVec2(p.x + w, p.y + h), col(cur ? C::accent : mix(C::border, C::border_hi, t), cur ? 0.8f : 1.f), px(12));
    float pad = px(16), s = px(36);
    ImVec2 ib(p.x + pad, p.y + pad);
    dl->AddRectFilled(ib, ImVec2(ib.x + s, ib.y + s), col(tint, 0.15f), px(9));
    icon(dl, IC_FOLDER, ImVec2(ib.x + s * 0.5f, ib.y + s * 0.5f), px(18), col(tint));
    ImFont* bf = F.bold ? F.bold : ImGui::GetFont();
    ImFont* sf = F.small ? F.small : ImGui::GetFont();
    float tx = ib.x + s + px(12), tw = p.x + w - pad - tx - px(30);
    text_ellipsis(dl, bf, ImVec2(tx, ib.y), tw, pi.name.c_str(), col(C::text));
    std::string when = ago(pi.updated);
    if (cur) when += "  \xC2\xB7  " + std::string(tr(S::ProjectCurrent));
    text_ellipsis(dl, sf, ImVec2(tx, ib.y + bf->FontSize + px(3)), tw, when.c_str(), col(cur ? C::good : C::muted));
    float y = ib.y + s + px(14);
    icon(dl, IC_DATA, ImVec2(p.x + pad + px(7), y + sf->FontSize * 0.5f), px(13), col(C::subtle));
    text_ellipsis(dl, sf, ImVec2(p.x + pad + px(20), y), w - pad * 2 - px(20), pi.data.empty() ? "-" : pi.data.c_str(), col(C::muted));
    y += sf->FontSize + px(6);
    std::string m = pi.model.empty() ? tr(S::ProjectNoModel) : strf(tr(S::ProjectModelFmt), fmt_count(pi.params).c_str());
    icon(dl, IC_MODEL, ImVec2(p.x + pad + px(7), y + sf->FontSize * 0.5f), px(13), col(C::subtle));
    text_ellipsis(dl, sf, ImVec2(p.x + pad + px(20), y), w - pad * 2 - px(20), m.c_str(), col(C::muted));
    if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    // folder button in the top right corner
    float fb = px(28);
    ImGui::SetCursorScreenPos(ImVec2(p.x + w - pad - fb + px(6), p.y + pad - px(4)));
    bool folder = ImGui::InvisibleButton("##folder", ImVec2(fb, fb));
    bool fh = ImGui::IsItemHovered();
    if (fh) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        tooltip(tr(S::ProjectsFolder));
    }
    ImVec2 fc(p.x + w - pad - fb * 0.5f + px(6), p.y + pad - px(4) + fb * 0.5f);
    if (fh) dl->AddRectFilled(ImVec2(fc.x - fb * 0.5f, fc.y - fb * 0.5f), ImVec2(fc.x + fb * 0.5f, fc.y + fb * 0.5f), col(C::surface3), px(7));
    icon(dl, IC_FOLDER, fc, px(14), col(fh ? C::text : C::subtle));
    ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + h));
    ImGui::Dummy(ImVec2(0, 0));
    if (folder) open_folder(pi.dir);
    else if (open) project_open_from_list(pi.dir);
}

void page_projects() {
    if (!A.projects_scanned) scan_projects();
    float bw = button_width(tr(S::OpenProject), IC_FOLDER) + px(10) + button_width(tr(S::NewProject), IC_PLUS);
    header(tr(S::NavProjects), tr(S::ProjectsHelp), bw);
    if (button(tr(S::OpenProject), BTN_SECONDARY, IC_FOLDER)) ask_open_project();
    ImGui::SameLine(0, px(10));
    if (button(tr(S::NewProject), BTN_PRIMARY, IC_PLUS)) ask_new_project();
    header_end();

    if (!A.project_msg.empty()) {
        if (A.project_ok) banner(A.project_msg.c_str(), C::good, IC_CHECK);
        else banner(strf(tr(S::ProjectFailed), A.project_msg.c_str()).c_str(), C::bad, IC_WARN);
        gap();
    }

    if (!A.project_dir.empty()) {
        begin_card("current");
        {
            FontScope f(F.h2);
            ImGui::TextUnformatted(A.project_name.c_str());
        }
        ImGui::SameLine(0, px(12));
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + px(3));
        badge(tr(S::ProjectCurrent), C::good);
        text(A.project_dir.c_str(), C::subtle);
        gap(8);
        std::string data = A.data.kind != DATA_NONE ? file_name(A.data.path) : "-";
        std::string m = A.model ? fmt_count((double)A.model->param_count(false)) + " " + tr(S::Params) : tr(S::ProjectNoModel);
        std::vector<Stat> st;
        st.push_back({tr(S::NavData), data, "", C::text, IC_DATA});
        st.push_back({tr(S::NavModel), m, "", A.model ? C::text : C::muted, IC_MODEL});
        st.push_back({tr(S::NavTest), std::to_string(std::count(A.tests, A.tests + std::strlen(A.tests), '\n') + (A.tests[0] ? 1 : 0)), "", C::text, IC_TARGET});
        stat_grid(st, px(170), false);
        gap(8);
        ImGui::BeginDisabled(busy());
        if (button(tr(S::ScSave), BTN_PRIMARY, IC_SAVE)) save_open_project();
        ImGui::EndDisabled();
        ImGui::SameLine(0, px(10));
        if (button(tr(S::ProjectsFolder), BTN_SECONDARY, IC_FOLDER)) open_folder(A.project_dir);
        gap(4);
        text_wrapped(tr(S::ProjectAuto), C::muted, F.small);
        end_card();
        gap(14);
    }

    if (A.projects.empty()) {
        int r = empty_state(IC_FOLDER, tr(S::NavProjects), tr(S::NoProjects), tr(S::NewProject), tr(S::OpenProject));
        if (r == 1) ask_new_project();
        if (r == 2) ask_open_project();
    } else {
        section_title(tr(S::RecentProjects));
        gap(2);
        float g = px(14), h = px(130);
        int cols = fit_columns(px(260), 3, g);
        float cw = column_width(cols, g);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(g, g));
        for (size_t i = 0; i < A.projects.size(); ++i) {
            if (i % cols) ImGui::SameLine(0, g);
            ImGui::PushID((int)i);
            ImGui::BeginGroup();
            project_tile(i, cw, h);
            ImGui::EndGroup();
            ImGui::PopID();
        }
        ImGui::PopStyleVar();
    }
    gap(14);
    std::string root = projects_root();
    text(strf("%s: %s", tr(S::ProjectsFolder), root.c_str()).c_str(), C::subtle);
    ImGui::SameLine(0, px(10));
    if (button(tr(S::Refresh), BTN_GHOST, IC_REFRESH)) A.projects_scanned = false;
}

void new_project_modal() {
    if (A.want_new_project) {
        A.want_new_project = false;
        ImGui::OpenPopup("##newproject");
    }
    ImGui::SetNextWindowSize(ImVec2(px(460), 0));
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(px(24), px(22)));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, px(14));
    bool open = ImGui::BeginPopupModal("##newproject", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    ImGui::PopStyleVar(2);
    if (!open) return;
    {
        FontScope f(F.h2);
        ImGui::TextUnformatted(tr(S::NewProject));
    }
    text_wrapped(tr(S::NewProjectHelp), C::muted, F.small);
    gap(8);
    text(tr(S::ProjectName), C::muted);
    if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
    ImGui::SetNextItemWidth(-FLT_MIN);
    bool enter = ImGui::InputText("##pname", A.new_project_name, sizeof A.new_project_name, ImGuiInputTextFlags_EnterReturnsTrue);
    gap(4);
    text(tr(S::ProjectDesc), C::muted);
    ImGui::SetNextItemWidth(-FLT_MIN);
    enter = enter |ImGui::InputText("##pdesc", A.new_project_desc, sizeof A.new_project_desc, ImGuiInputTextFlags_EnterReturnsTrue);
    if (!A.new_project_err.empty()) text_wrapped(A.new_project_err.c_str(), C::bad, F.small);
    gap(10);
    bool create = button(tr(S::CreateBtn), BTN_PRIMARY, IC_PLUS) || enter;
    ImGui::SameLine(0, px(10));
    bool cancel = button(tr(S::CancelBtn), BTN_GHOST) || ImGui::IsKeyPressed(ImGuiKey_Escape);
    if (create) {
        std::string err;
        if (create_project(A.new_project_name, A.new_project_desc, err)) {
            A.page = PAGE_PROJECTS;
            ImGui::CloseCurrentPopup();
        } else {
            A.new_project_err = err;
        }
    } else if (cancel) {
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

// Ctrl+K: jump to a page or run an action by typing part of its name.
void command_palette() {
    if (A.want_palette) {
        A.want_palette = false;
        A.palette_q[0] = 0;
        A.palette_sel = 0;
        ImGui::OpenPopup("##palette");
    }
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x * 0.5f, vp->WorkPos.y + vp->WorkSize.y * 0.16f), ImGuiCond_Always, ImVec2(0.5f, 0));
    ImGui::SetNextWindowSize(ImVec2(std::min(px(560), vp->WorkSize.x - px(40)), 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(px(12), px(12)));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, px(14));
    ImGui::PushStyleColor(ImGuiCol_Border, C::border_hi);
    bool open = ImGui::BeginPopupModal("##palette", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
    if (!open) return;

    struct Item { const char* label; Icon ic; int page; int action; };
    static const S page_labels[PAGE_COUNT] = {S::NavHome, S::NavProjects, S::NavData, S::NavModel, S::NavTrain, S::NavTest, S::NavChat, S::NavFiles, S::NavPlugins};
    static const Icon page_icons[PAGE_COUNT] = {IC_HOME, IC_FOLDER, IC_DATA, IC_MODEL, IC_TRAIN, IC_CHAT, IC_CHATS, IC_FILES, IC_PLUGIN};
    std::vector<Item> all;
    for (int i = 0; i < PAGE_COUNT; ++i) all.push_back({tr(page_labels[i]), page_icons[i], i, 0});
    all.push_back({tr(S::OpenProject), IC_FOLDER, -1, 1});
    all.push_back({tr(S::NewProject), IC_PLUS, -1, 2});
    if (!A.project_dir.empty()) all.push_back({tr(S::ScSave), IC_SAVE, -1, 3});
    all.push_back({tr(S::QaImport), IC_UPLOAD, PAGE_DATA, 0});
    all.push_back({tr(S::ChatWithModel), IC_CHATS, PAGE_CHAT, 0});
    all.push_back({tr(S::Terminal), IC_TERMINAL, -1, 4});
    all.push_back({tr(S::Settings), IC_GEAR, -1, 5});

    auto lower = [](std::string s) {
        for (char& ch : s) ch = (char)std::tolower((unsigned char)ch);
        return s;
    };
    std::string q = lower(A.palette_q);
    std::vector<const Item*> shown;
    for (const Item& it : all)
        if ((q.empty() || lower(it.label).find(q) != std::string::npos) && shown.size() < 9) shown.push_back(&it);

    if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::InputTextWithHint("##q", tr(S::PaletteHint), A.palette_q, sizeof A.palette_q)) A.palette_sel = 0;
    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) ++A.palette_sel;
    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) --A.palette_sel;
    A.palette_sel = shown.empty() ? 0 : std::max(0, std::min((int)shown.size() - 1, A.palette_sel));
    bool enter = ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter);
    gap(4);
    const Item* run = nullptr;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(px(8), px(2)));
    for (size_t i = 0; i < shown.size(); ++i) {
        ImGui::PushID((int)i);
        bool sel = (int)i == A.palette_sel;
        if (sel) {
            ImVec2 p = ImGui::GetCursorScreenPos();
            ImGui::GetWindowDrawList()->AddRectFilled(p, ImVec2(p.x + ImGui::GetContentRegionAvail().x, p.y + px(40)), col(C::accent, 0.16f), px(8));
        }
        if (list_row("##it", shown[i]->ic, sel ? C::accent_hi : C::muted, shown[i]->label, nullptr, sel ? "Enter" : "")) run = shown[i];
        ImGui::PopID();
    }
    ImGui::PopStyleVar();
    if (shown.empty()) text(tr(S::NoMatches), C::muted);
    if (enter && !shown.empty()) run = shown[A.palette_sel];
    if (run || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
    if (!run) return;
    if (run->page >= 0) A.page = run->page;
    if (run->action == 1) ask_open_project();
    if (run->action == 2) ask_new_project();
    if (run->action == 3) save_open_project();
    if (run->action == 4) open_terminal_here();
    if (run->action == 5) A.want_settings = true;
}

void shortcuts() {
    ImGuiIO& io = ImGui::GetIO();
    if (!io.KeyCtrl || io.KeyShift || io.KeyAlt) return;
    if (ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId)) return;
    if (ImGui::IsKeyPressed(ImGuiKey_K, false)) A.want_palette = true;
    if (ImGui::IsKeyPressed(ImGuiKey_S, false) && !busy()) save_or_create_project();
    if (ImGui::IsKeyPressed(ImGuiKey_O, false)) ask_open_project();
    if (ImGui::IsKeyPressed(ImGuiKey_GraveAccent, false)) open_terminal_here();
    if (ImGui::IsKeyPressed(ImGuiKey_Comma, false)) A.want_settings = true;
}

void settings_modal() {
    if (A.want_settings) {
        A.want_settings = false;
        ImGui::OpenPopup("##settings");
    }
    ImGui::SetNextWindowSize(ImVec2(px(480), 0));
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(px(24), px(22)));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, px(14));
    bool open = ImGui::BeginPopupModal("##settings", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    ImGui::PopStyleVar(2);
    if (!open) return;
    {
        FontScope f(F.h2);
        ImGui::TextUnformatted(tr(S::Settings));
    }
    gap(8);
    field_label(tr(S::Language), px(130));
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (begin_combo("##lang", language_name(get_language()))) {
        for (int l = 0; l < LANG_COUNT; ++l)
            if (ImGui::Selectable(language_name(l), l == get_language())) set_language(l);
        ImGui::EndCombo();
    }
    gap(10);
    float bw = ImGui::GetContentRegionAvail().x;
    if (action_row(tr(S::OpenConfigDir), IC_GEAR)) open_folder(config_dir());
    if (action_row(tr(S::OpenModelsDir), IC_CUBE)) {
        std::error_code ec;
        std::filesystem::create_directories(upath(models_dir()), ec);
        open_folder(models_dir());
    }
    if (action_row(tr(S::ProjectsFolder), IC_FOLDER)) {
        std::error_code ec;
        std::filesystem::create_directories(upath(projects_root()), ec);
        open_folder(projects_root());
    }
    if (action_row(tr(S::Terminal), IC_TERMINAL)) open_terminal_here();
    gap(10);
    {
        FontScope f(F.bold);
        ImGui::TextUnformatted(tr(S::Updates));
    }
    ImGui::Checkbox(tr(S::UpdateAtStart), &A.update_check);
    int us = A.update->state;
    if (us == 1) text(tr(S::UpdateChecking), C::muted);
    else if (us == 3) text_wrapped(strf(tr(S::UpdateFailed), A.update->err.c_str()).c_str(), C::warn, F.small);
    else if (us == 2 && update_ready()) {
        text(strf(tr(S::UpdateAvailable), A.update->latest.c_str()).c_str(), C::accent_hi);
        ImGui::SameLine();
        if (button(tr(S::UpdateDownload), BTN_PRIMARY)) open_folder(A.update->url);
    } else if (us == 2) text(tr(S::UpdateLatest), C::good);
    if (us != 1 && button(tr(S::UpdateNow), BTN_SECONDARY, IC_REFRESH)) start_update_check();
    gap(10);
    text_wrapped(strf(tr(S::AboutLine), KOR_VERSION).c_str(), C::muted, F.small);
    gap(10);
    if (button("OK", BTN_PRIMARY, IC_CHECK, bw) || ImGui::IsKeyPressed(ImGuiKey_Escape)) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

// ------------------------------------------------------------------- data

void clean_preview() { A.clean_out = clean_text(A.clean_src, A.q_tag, A.a_tag, A.clean_opt, A.clean_rep); }

void clean_modal() {
    if (A.want_clean) {
        A.want_clean = false;
        if (!read_whole(A.data.path, A.clean_src)) {
            A.data_ok = false;
            A.data_msg = "cannot open file";
            return;
        }
        clean_preview();
        ImGui::OpenPopup("##clean");
    }
    ImGui::SetNextWindowSize(ImVec2(px(520), 0));
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(px(24), px(22)));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, px(14));
    bool open = ImGui::BeginPopupModal("##clean", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    ImGui::PopStyleVar(2);
    if (!open) return;
    {
        FontScope f(F.h2);
        ImGui::TextUnformatted(tr(S::CleanData));
    }
    std::string stem = upath(A.data.path).stem().u8string();
    if (stem.size() > 6 && stem.compare(stem.size() - 6, 6, "-clean") == 0) stem.resize(stem.size() - 6);
    text_wrapped(strf(tr(S::CleanHelp), (stem + "-clean.txt").c_str()).c_str(), C::muted, F.small);
    gap(8);
    CleanOptions& o = A.clean_opt;
    bool changed = ImGui::Checkbox(tr(S::CleanSpaces), &o.trim);
    o.collapse_spaces = o.trim;
    changed |= ImGui::Checkbox(tr(S::CleanEmpty), &o.drop_empty);
    changed |= ImGui::Checkbox(tr(S::CleanDupes), &o.dedupe);
    tooltip(tr(S::CleanDupesHelp));
    if (!A.q_tag.empty()) changed |= ImGui::Checkbox(tr(S::CleanBroken), &o.drop_broken);
    if (changed) clean_preview();
    gap(8);
    const CleanReport& r = A.clean_rep;
    bool nothing = r.duplicates + r.empty + r.broken + r.changed == 0;
    if (nothing) banner(tr(S::CleanNothing), C::good, IC_CHECK);
    else
        banner(strf(tr(S::CleanPreview), (int)r.lines_in, (int)r.lines_out, (int)r.duplicates, (int)r.empty, (int)r.broken, (int)r.changed).c_str(),
               C::accent_hi, IC_INFO);
    gap(10);
    ImGui::BeginDisabled(nothing);
    bool go = button(tr(S::CleanLoad), BTN_PRIMARY, IC_CHECK);
    ImGui::EndDisabled();
    ImGui::SameLine(0, px(10));
    bool cancel = button(tr(S::CancelBtn), BTN_GHOST) || ImGui::IsKeyPressed(ImGuiKey_Escape);
    if (go) {
        CleanReport done = r;
        std::string dst = write_derived(A.data.path, "-clean.txt", A.clean_out);
        if (dst.empty()) {
            A.data_ok = false;
            A.data_msg = strf(tr(S::CantWrite), (stem + "-clean.txt").c_str());
        } else if (load_data_impl(dst, 0, -1)) {
            A.data_note = strf(tr(S::CleanDone), (int)done.duplicates, (int)(done.empty + done.broken), (int)done.changed, file_name(dst).c_str());
        }
    }
    if (go || cancel) {
        A.clean_src.clear();
        A.clean_out.clear();
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

void page_data() {
    header(tr(S::NavData), tr(S::DataHelp));
    header_end();
    clean_modal();

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
                std::string ext = upath(p).extension().u8string();
                for (char& ch : ext) ch = (char)std::tolower((unsigned char)ch);
                if (ext == ".csv" || ext == ".tsv") A.data_kind = 1;
                else if (ext == ".txt" || ext == ".jsonl" || ext == ".json") A.data_kind = 0;
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
    if (A.data_kind == 0) {
        ImGui::AlignTextToFramePadding();
        text(tr(S::Tools), C::muted);
        ImGui::SameLine();
        if (has_native_dialogs()) {
            if (button(tr(S::ImportJsonl), BTN_GHOST, IC_UPLOAD)) {
                std::string p = open_file_dialog(DLG_DATA);
                if (!p.empty()) {
                    std::snprintf(A.data_path, sizeof A.data_path, "%s", p.c_str());
                    load_data_impl(p, 0, -1);
                }
            }
            tooltip(tr(S::ImportJsonlHelp));
            ImGui::SameLine(0, px(4));
        }
        ImGui::BeginDisabled(A.data.kind != DATA_TEXT);
        if (button(tr(S::CleanData), BTN_GHOST, IC_SPARK)) A.want_clean = true;
        ImGui::EndDisabled();
    }
    ImGui::EndDisabled();
    if (A.data_ok && !A.data_note.empty()) banner(A.data_note.c_str(), C::good, IC_CHECK);
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
    if (A.data.kind == DATA_TEXT) {
        ImGui::SameLine(0, px(28));
        checkbox(tr(S::UseGpu), &A.use_gpu);
        tooltip(tr(S::UseGpuHelp));
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

void compare_view(float h);

void page_train() {
    if (A.data.kind == DATA_NONE) {
        header(tr(S::NavTrain), tr(S::TrainHelp));
        header_end();
        resume_card();
        if (!A.runs_scanned) scan_runs();
        if (A.runs.empty()) {
            if (empty_state(IC_DATA, nullptr, tr(S::NoData), tr(S::GoData)) == 1) A.page = PAGE_DATA;
            return;
        }
        banner(tr(S::NoData), C::blue, IC_INFO);
        if (button(tr(S::GoData), BTN_SECONDARY, IC_DATA)) A.page = PAGE_DATA;
        gap(8);
        section_title(tr(S::TabCompare));
        compare_view(std::max(px(300), ImGui::GetContentRegionAvail().y - px(2)));
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
    resume_card();

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
    if (!A.cfg.checkpoint.empty() && (run || st.state == TS_STOPPED)) {
        gap(2);
        if (!st.save_error.empty()) text_wrapped(strf(tr(S::AutosaveFailed), st.save_error.c_str()).c_str(), C::bad, F.small);
        else text_wrapped(st.saved_step > 0 ? strf(tr(S::AutosaveLbl), st.saved_step).c_str() : tr(S::AutosaveWait), C::subtle, F.small);
        tooltip(tr(S::AutosaveHelp));
    }
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
    labels.push_back(tr(S::TabCompare));
    ids.push_back(4);
    int sel = 0;
    for (int i = 0; i < (int)ids.size(); ++i)
        if (ids[i] == A.train_tab) sel = i;
    tabs("traintabs", &sel, labels.data(), (int)labels.size());
    A.train_tab = ids[sel];
    float h = std::max(px(300), ImGui::GetContentRegionAvail().y - px(2));
    if (A.train_tab == 0) plot_loss(st, ImVec2(ImGui::GetContentRegionAvail().x, h));
    else if (A.train_tab == 1) log_view(h);
    else if (A.train_tab == 2) tests_view(st);
    else if (A.train_tab == 3) skill_metrics(st);
    else compare_view(h);
}

// ------------------------------------------------------------- comparison

const ImVec4& run_color(int i) {
    static const ImVec4* palette[] = {&C::accent_hi, &C::blue, &C::good, &C::warn, &C::violet, &C::teal, &C::bad};
    return *palette[i % 7];
}

void compare_chart(const std::vector<const State::RunRecord*>& sel, const std::vector<int>& colors, bool val, ImVec2 size) {
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##cmpplot", size);
    bool hov = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p1(p0.x + size.x, p0.y + size.y);
    dl->AddRectFilled(p0, p1, col(C::surface), px(12));
    dl->AddRect(p0, p1, col(C::border), px(12));
    ImFont* sf = F.small ? F.small : ImGui::GetFont();
    float lo = 1e30f, hi = -1e30f;
    int total = 2;
    for (const State::RunRecord* r : sel) {
        const std::vector<float>& v = val ? r->val_hist : r->train_hist;
        for (float x : v) { lo = std::min(lo, x); hi = std::max(hi, x); }
        const std::vector<int>& s = val ? r->val_steps : r->train_steps;
        if (!s.empty()) total = std::max(total, s.back());
    }
    if (lo > hi) {
        const char* msg = tr(S::PlotEmpty);
        ImVec2 ts = ImGui::CalcTextSize(msg);
        dl->AddText(ImVec2(p0.x + (size.x - ts.x) * 0.5f, p0.y + (size.y - ts.y) * 0.5f), col(C::muted), msg);
        return;
    }
    if (hi - lo < 1e-6f) hi = lo + 1e-6f;
    float pad = (hi - lo) * 0.06f;
    lo = std::max(0.f, lo - pad);
    hi += pad;
    ImVec2 a(p0.x + px(60), p0.y + px(20)), b(p1.x - px(22), p1.y - px(36));
    auto X = [&](float step) { return a.x + (b.x - a.x) * (step - 1) / std::max(1, total - 1); };
    auto Y = [&](float v) { return b.y - (b.y - a.y) * (v - lo) / (hi - lo); };
    char t[64];
    for (int gi = 0; gi <= 4; ++gi) {
        float v = lo + (hi - lo) * gi / 4.f, y = Y(v);
        dl->AddLine(ImVec2(a.x, y), ImVec2(b.x, y), col(C::border, gi == 0 ? 1.f : 0.55f));
        std::snprintf(t, sizeof t, "%.3f", v);
        ImVec2 ts = text_size(sf, t);
        dl->AddText(sf, sf->FontSize, ImVec2(a.x - ts.x - px(10), y - ts.y * 0.5f), col(C::subtle), t);
        int step = gi == 0 ? 1 : (int)std::lround((double)total * gi / 4);
        std::snprintf(t, sizeof t, "%d", step);
        ts = text_size(sf, t);
        float x = std::min(std::max(X((float)step) - ts.x * 0.5f, a.x - px(4)), b.x - ts.x + px(4));
        dl->AddText(sf, sf->FontSize, ImVec2(x, b.y + px(10)), col(C::subtle), t);
    }
    for (size_t k = 0; k < sel.size(); ++k) {
        const std::vector<float>& v = val ? sel[k]->val_hist : sel[k]->train_hist;
        const std::vector<int>& s = val ? sel[k]->val_steps : sel[k]->train_steps;
        std::vector<ImVec2> pts;
        for (size_t i = 0; i < v.size() && i < s.size(); ++i) pts.push_back(ImVec2(X((float)s[i]), Y(v[i])));
        ImU32 c = col(run_color(colors[k]));
        if (pts.size() >= 2) dl->AddPolyline(pts.data(), (int)pts.size(), c, 0, px(2.f));
        if (val)
            for (const ImVec2& q : pts) dl->AddCircleFilled(q, px(2.5f), c);
    }
    ImVec2 m = ImGui::GetIO().MousePos;
    if (hov && m.x >= a.x && m.x <= b.x) {
        float step = (m.x - a.x) / std::max(1.f, b.x - a.x) * (total - 1) + 1;
        dl->AddLine(ImVec2(m.x, a.y), ImVec2(m.x, b.y), col(C::border_hi), 1.f);
        ImGui::BeginTooltip();
        ImGui::Text(tr(S::StepLbl), (int)std::lround(step));
        for (size_t k = 0; k < sel.size(); ++k) {
            const std::vector<float>& v = val ? sel[k]->val_hist : sel[k]->train_hist;
            const std::vector<int>& s = val ? sel[k]->val_steps : sel[k]->train_steps;
            if (v.empty()) continue;
            size_t best = 0;
            for (size_t i = 1; i < s.size(); ++i)
                if (std::abs(s[i] - step) < std::abs(s[best] - step)) best = i;
            ImGui::TextColored(run_color(colors[k]), "%s  %.4f", sel[k]->name.c_str(), v[best]);
        }
        ImGui::EndTooltip();
    }
}

void compare_view(float h) {
    if (!A.runs_scanned) scan_runs();
    if (A.runs.empty()) {
        empty_state(IC_TREND, nullptr, tr(S::CompareEmpty));
        return;
    }
    static int metric = 0;  // 0 validation loss, 1 training loss
    std::vector<const State::RunRecord*> sel;
    std::vector<int> colors;
    for (size_t i = 0; i < A.runs.size(); ++i)
        if (A.runs[i].selected) {
            sel.push_back(&A.runs[i]);
            colors.push_back((int)i);
        }
    const char* metrics[] = {tr(S::ValLoss), tr(S::Loss)};
    segmented("cmpmetric", &metric, metrics, 2);
    ImGui::SameLine(0, px(14));
    text(tr(S::CompareHint), C::subtle);
    gap(4);
    compare_chart(sel, colors, metric == 0, ImVec2(ImGui::GetContentRegionAvail().x, std::max(px(240), h * 0.45f)));
    gap(8);

    // the best value per column among all records
    auto best_of = [](auto get, bool low) {
        double b = low ? 1e30 : -1e30;
        for (auto& r : A.runs) {
            double v = get(r);
            if (v >= 0) b = low ? std::min(b, v) : std::max(b, v);
        }
        return b;
    };
    double b_val = best_of([](const State::RunRecord& r) { return (double)r.best_val; }, true);
    double b_loss = best_of([](const State::RunRecord& r) { return (double)r.loss; }, true);
    double b_acc = best_of([](const State::RunRecord& r) { return (double)r.val_acc; }, false);
    double b_tests = best_of([](const State::RunRecord& r) { return r.tests_total > 0 ? (double)r.tests_passed / r.tests_total : -1.0; }, false);

    ImGuiTableFlags tf = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp |
                         ImGuiTableFlags_PadOuterX | ImGuiTableFlags_ScrollY;
    ImGui::PushStyleColor(ImGuiCol_TableRowBg, C::surface);
    ImGui::PushStyleColor(ImGuiCol_TableRowBgAlt, mix(C::surface, C::surface2, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_TableBorderLight, C::border);
    float th = std::max(px(180), ImGui::GetContentRegionAvail().y - px(4));
    if (ImGui::BeginTable("##runs", 11, tf, ImVec2(0, th))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, px(46));
        ImGui::TableSetupColumn(tr(S::ColData), 0, 2.2f);
        ImGui::TableSetupColumn(tr(S::ColDate), 0, 1.4f);
        ImGui::TableSetupColumn(tr(S::Steps), 0, 1.0f);
        ImGui::TableSetupColumn(tr(S::Elapsed), 0, 1.0f);
        ImGui::TableSetupColumn(tr(S::NavModel), 0, 1.6f);
        ImGui::TableSetupColumn(tr(S::Loss), 0, 1.1f);
        ImGui::TableSetupColumn(tr(S::ColBestVal), 0, 1.3f);
        ImGui::TableSetupColumn(tr(S::Accuracy), 0, 1.0f);
        ImGui::TableSetupColumn(tr(S::TabTests), 0, 1.0f);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, px(30));
        ImGui::TableHeadersRow();
        int remove = -1;
        for (size_t i = 0; i < A.runs.size(); ++i) {
            State::RunRecord& r = A.runs[i];
            ImGui::PushID((int)i);
            ImGui::TableNextRow(0, px(30));
            ImGui::TableNextColumn();
            ImGui::Checkbox("##sel", &r.selected);
            ImGui::SameLine(0, px(4));
            ImVec2 cp = ImGui::GetCursorScreenPos();
            float fh = ImGui::GetFrameHeight();
            ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(cp.x + px(5), cp.y + fh * 0.5f), px(4), col(run_color((int)i), r.selected ? 1.f : 0.3f));
            ImGui::Dummy(ImVec2(px(10), fh));
            auto cell = [&](const std::string& s, bool best = false, const ImVec4& c = C::text) {
                ImGui::TableNextColumn();
                ImGui::AlignTextToFramePadding();
                ImGui::TextColored(best ? C::good : c, "%s", s.c_str());
            };
            cell(r.name.empty() ? "-" : r.name);
            if (ImGui::IsItemHovered() && !r.state.empty() && r.state != "done") tooltip(tr(S::RunStopped));
            char when[32];
            std::strftime(when, sizeof when, "%d.%m %H:%M", std::localtime(&r.time));
            cell(when, false, C::muted);
            cell(r.state == "done" ? std::to_string(r.steps) : strf("%d / %d", r.steps, r.total), false, r.state == "done" ? C::text : C::warn);
            cell(fmt_time(r.elapsed), false, C::muted);
            cell(r.arch + "  \xC2\xB7  " + fmt_count(r.params), false, C::muted);
            cell(r.loss >= 0 ? strf("%.4f", r.loss) : "-", r.loss >= 0 && r.loss == (float)b_loss);
            cell(r.best_val >= 0 ? strf("%.4f", r.best_val) : "-", r.best_val >= 0 && r.best_val == (float)b_val);
            cell(r.val_acc >= 0 ? strf("%.1f%%", r.val_acc * 100) : "-", r.val_acc >= 0 && r.val_acc == (float)b_acc);
            bool tests = r.tests_total > 0 && r.tests_passed >= 0;
            cell(tests ? strf("%d/%d", r.tests_passed, r.tests_total) : "-", tests && (double)r.tests_passed / r.tests_total == b_tests);
            ImGui::TableNextColumn();
            {
                ImVec2 bp = ImGui::GetCursorScreenPos();
                float bs = ImGui::GetFrameHeight();
                if (ImGui::InvisibleButton("##del", ImVec2(bs, bs))) remove = (int)i;
                bool dh = ImGui::IsItemHovered();
                if (dh) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    ImGui::GetWindowDrawList()->AddRectFilled(bp, ImVec2(bp.x + bs, bp.y + bs), col(C::bad, 0.15f), px(6));
                }
                icon(ImGui::GetWindowDrawList(), IC_TRASH, ImVec2(bp.x + bs * 0.5f, bp.y + bs * 0.5f), px(14), col(dh ? C::bad : C::subtle));
                tooltip(tr(S::DeleteRun));
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
        if (remove >= 0) {
            std::error_code ec;
            std::filesystem::remove(upath(A.runs[remove].file), ec);
            A.runs.erase(A.runs.begin() + remove);
        }
    }
    ImGui::PopStyleColor(3);
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
        if (A.file_ok) add_activity(State::ACT_MODEL_SAVED, file_name(A.model_path));
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

// ------------------------------------------------------------------- chat

bool chat_ready() { return A.chat.model && A.chat.model->spec.kind == MODEL_GPT && A.chat.model->spec.media == MEDIA_TEXT; }

void chat_stop() {
    A.chat.cancel = true;
    if (A.chat.th.joinable()) A.chat.th.join();
    A.chat.cancel = false;
    A.chat.running = false;
}

// Moves a finished answer into the conversation; called every frame.
void chat_collect() {
    State::Chat& c = A.chat;
    if (c.running || !c.th.joinable()) return;
    c.th.join();
    std::string answer;
    {
        std::lock_guard<std::mutex> lk(c.mu);
        answer.swap(c.reply);
    }
    size_t a = answer.find_first_not_of(" \t"), b = answer.find_last_not_of(" \t\n");
    answer = a == std::string::npos ? "" : answer.substr(a, b - a + 1);
    c.msgs.push_back({false, answer.empty() ? "..." : answer});
    c.scroll = true;
}

void remember_chat_model(const std::string& path) {
    auto& r = A.chat.recent;
    r.erase(std::remove(r.begin(), r.end(), path), r.end());
    r.insert(r.begin(), path);
    if (r.size() > 8) r.resize(8);
}

void scan_chat_models() {
    namespace fs = std::filesystem;
    State::Chat& c = A.chat;
    c.files.clear();
    c.scanned = true;
    std::set<std::string> seen;
    auto add = [&](const fs::path& p) {
        std::error_code ec;
        if (!fs::is_regular_file(p, ec)) return;
        std::string key = fs::weakly_canonical(p, ec).u8string();
        if (!seen.insert(key).second) return;
        State::Chat::File f;
        f.path = p.u8string();
        f.name = p.filename().u8string();
        f.mb = (double)fs::file_size(p, ec) / (1024.0 * 1024.0);
        auto ft = fs::last_write_time(p, ec);
        if (!ec) {
            auto sys = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                ft - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
            std::time_t t = std::chrono::system_clock::to_time_t(sys);
            std::tm tm = *std::localtime(&t);
            char b[32];
            std::strftime(b, sizeof b, "%Y-%m-%d %H:%M", &tm);
            f.when = b;
        }
        c.files.push_back(f);
    };
    std::error_code ec;
    for (fs::directory_iterator it(upath(models_dir()), ec), end; !ec && it != end; it.increment(ec)) {
        std::string ext = it->path().extension().u8string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) { return (char)std::tolower(ch); });
        if (ext == ".kor") add(it->path());
    }
    std::sort(c.files.begin(), c.files.end(), [](const State::Chat::File& x, const State::Chat::File& y) { return x.when > y.when; });
    for (const std::string& r : c.recent) add(upath(r));
}

void chat_set_model(std::shared_ptr<Model> m, const std::string& path, const std::string& name) {
    chat_stop();
    State::Chat& c = A.chat;
    c.model = m;
    c.path = path;
    c.name = name;
    c.msgs.clear();
    c.queue.clear();
    c.reply.clear();
    c.focus = true;
    // Dialogue tags: stored in the model by this version, else those of the loaded data, else the usual ones.
    std::string q = "Kullanıcı: ", a = "Asistan: ";
    auto qi = m->meta.find("q_tag"), ai = m->meta.find("a_tag");
    if (qi != m->meta.end() && ai != m->meta.end()) { q = qi->second; a = ai->second; }
    else if (!A.q_tag.empty() && !A.a_tag.empty()) { q = A.q_tag; a = A.a_tag; }
    std::snprintf(c.q_tag, sizeof c.q_tag, "%s", q.c_str());
    std::snprintf(c.a_tag, sizeof c.a_tag, "%s", a.c_str());
}

bool chat_open(const std::string& path) {
    std::string err;
    auto m = Model::load(path, &err);
    A.chat.ok = (bool)m;
    A.chat.msg = m ? "" : err;
    if (!m) return false;
    chat_set_model(m, path, upath(path).filename().u8string());
    if (path.find(models_dir()) != 0) remember_chat_model(path);
    A.chat.scanned = false;
    return true;
}

// A copy of the model being trained, so the chat never waits for training.
void chat_use_current() {
    if (!A.model) return;
    std::string snap = (upath(config_dir()) / "chat-model.tmp").u8string(), err;
    bool saved;
    {
        std::lock_guard<std::mutex> lk(A.trainer.model_mu);
        saved = A.model->save(snap, false, &err);
    }
    auto m = saved ? Model::load(snap, &err) : nullptr;
    std::error_code ec;
    std::filesystem::remove(upath(snap), ec);
    A.chat.ok = (bool)m;
    A.chat.msg = m ? "" : err;
    if (!m) return;
    if (!A.q_tag.empty() && !A.a_tag.empty()) {
        m->meta["q_tag"] = A.q_tag;
        m->meta["a_tag"] = A.a_tag;
    }
    chat_set_model(m, "", tr(S::ChatUseCurrent));
}

// The prompt for the newest message: as much of the conversation as fits in
// the model's context, newest last, in the format of the training text.
std::vector<int> chat_prompt() {
    State::Chat& c = A.chat;
    const Model& m = *c.model;
    std::string q = c.q_tag, a = c.a_tag;
    bool tagged = !q.empty() || !a.empty();
    const std::string& last = c.msgs.back().text;
    std::vector<int> ids = tokenize(tagged ? q + last + "\n" + a : last, m.vocab);
    if (!tagged || !c.memory || ids.empty()) return ids;
    int budget = std::max(8, m.spec.ctx - std::min(c.max_len, m.spec.ctx / 2)) - (int)ids.size();
    for (int i = (int)c.msgs.size() - 2; i >= 0 && budget > 0; --i) {
        const State::Chat::Msg& msg = c.msgs[i];
        std::vector<int> part = tokenize((msg.user ? q : a) + msg.text + "\n", m.vocab);
        if ((int)part.size() > budget) break;
        budget -= (int)part.size();
        ids.insert(ids.begin(), part.begin(), part.end());
    }
    return ids;
}

void chat_send(const std::string& text) {
    State::Chat& c = A.chat;
    std::string t = text;
    while (!t.empty() && (t.back() == ' ' || t.back() == '\n')) t.pop_back();
    if (t.empty() || !chat_ready()) return;
    if (c.running || c.th.joinable()) { c.queue.push_back(t); return; }
    c.msgs.push_back({true, t});
    c.scroll = true;
    std::vector<int> ids = chat_prompt();
    if (ids.empty()) {
        c.msgs.push_back({false, tr(S::ChatUnknown)});
        return;
    }
    int newline = -1;
    for (size_t i = 0; i < c.model->vocab.size(); ++i)
        if (c.model->vocab[i] == "\n") newline = (int)i;
    bool tagged = c.q_tag[0] || c.a_tag[0];
    {
        std::lock_guard<std::mutex> lk(c.mu);
        c.reply.clear();
    }
    c.running = true;
    c.cancel = false;
    std::shared_ptr<Model> m = c.model;
    int n = std::max(1, c.max_len);
    float temp = c.temperature;
    uint64_t seed = c.rng.next();
    c.th = std::thread([m, ids, n, temp, seed, newline, tagged] {
        State::Chat& ch = A.chat;
        Rng rng(seed);
        // Dialogue models answer in one line; plain text just continues.
        m->generate(ids, n, temp, rng, tagged ? newline : -1, [&](int t) {
            if (t == newline && tagged) return false;
            if (t >= 0 && t < (int)m->vocab.size()) {
                std::lock_guard<std::mutex> lk(ch.mu);
                ch.reply += m->vocab[t];
            }
            return !ch.cancel.load();
        });
        ch.running = false;
    });
}

// One message as a bubble: the user's on the right, the model's on the left.
void chat_bubble(const std::string& text, bool user, bool typing) {
    float avail = ImGui::GetContentRegionAvail().x;
    float wrap = std::min(avail * 0.78f, px(620));
    ImFont* font = ImGui::GetFont();
    float fs = ImGui::GetFontSize();
    std::string shown = text;
    if (typing) {
        int dots = 1 + (int)(ImGui::GetTime() * 3.0) % 3;
        shown += shown.empty() ? std::string(dots, '.') : std::string(" ") + std::string(dots, '.');
    }
    ImVec2 ts = font->CalcTextSizeA(fs, FLT_MAX, wrap, shown.c_str());
    float padx = px(14), pady = px(9);
    ImVec2 size(std::max(ts.x, px(24)) + padx * 2, ts.y + pady * 2);
    ImFont* lf = F.small ? F.small : font;
    const char* who = tr(user ? S::ChatUser : S::ChatBot);
    ImVec2 p = ImGui::GetCursorScreenPos();
    float x = user ? p.x + avail - size.x : p.x;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    float lw = text_size(lf, who).x;
    dl->AddText(lf, lf->FontSize, ImVec2(user ? p.x + avail - lw - px(4) : p.x + px(4), p.y), col(C::subtle), who);
    float y = p.y + lf->FontSize + px(3);
    ImU32 bg = user ? col(C::accent, 0.22f) : col(C::surface2);
    dl->AddRectFilled(ImVec2(x, y), ImVec2(x + size.x, y + size.y), bg, px(12));
    dl->AddRect(ImVec2(x, y), ImVec2(x + size.x, y + size.y), user ? col(C::accent, 0.45f) : col(C::border), px(12));
    dl->AddText(font, fs, ImVec2(x + padx, y + pady), col(typing && text.empty() ? C::muted : C::text), shown.c_str(), nullptr, wrap);
    ImGui::SetCursorScreenPos(ImVec2(x, y));
    ImGui::InvisibleButton("##bubble", size);
    if (ImGui::IsItemClicked(ImGuiMouseButton_Right) && !text.empty()) ImGui::SetClipboardText(text.c_str());
    ImGui::SetCursorScreenPos(ImVec2(p.x, y + size.y + px(10)));
    ImGui::Dummy(ImVec2(0, 0));
}

void chat_model_list() {
    State::Chat& c = A.chat;
    if (!c.scanned) scan_chat_models();
    begin_card("chatmodels", tr(S::ChatModels));
    text_wrapped(tr(S::ChatModelsHelp), C::muted, F.small);
    gap(2);
    if (c.files.empty()) {
        text_wrapped(tr(S::ChatNoModels), C::subtle);
    } else {
        float row_h = ImGui::GetFrameHeight();
        float h = std::min((float)c.files.size(), 5.f) * (row_h + px(4)) + px(26);
        if (begin_inset("##chatfiles", h)) {
            for (size_t i = 0; i < c.files.size(); ++i) {
                const State::Chat::File& f = c.files[i];
                ImGui::PushID((int)i);
                bool sel = !c.path.empty() && f.path == c.path;
                ImVec2 p = ImGui::GetCursorScreenPos();
                float w = ImGui::GetContentRegionAvail().x;
                if (ImGui::Selectable("##f", sel, 0, ImVec2(w, row_h)) && !sel) chat_open(f.path);
                ImDrawList* dl = ImGui::GetWindowDrawList();
                float ty = p.y + (row_h - ImGui::GetFontSize()) * 0.5f;
                icon(dl, IC_MODEL, ImVec2(p.x + px(10), p.y + row_h * 0.5f), px(15), col(sel ? C::accent : C::muted));
                std::string info = f.when + "   " + strf("%.1f MB", f.mb);
                ImFont* sf = F.small ? F.small : ImGui::GetFont();
                float iw = text_size(sf, info.c_str()).x;
                text_ellipsis(dl, ImGui::GetFont(), ImVec2(p.x + px(26), ty), w - iw - px(40), f.name.c_str(), col(sel ? C::accent_hi : C::text));
                dl->AddText(sf, sf->FontSize, ImVec2(p.x + w - iw - px(4), p.y + (row_h - sf->FontSize) * 0.5f), col(C::subtle), info.c_str());
                if (ImGui::IsItemHovered()) tooltip(f.path.c_str());
                ImGui::PopID();
            }
        }
        ImGui::EndChild();
    }
    gap(4);
    if (has_native_dialogs()) {
        if (button(tr(S::Browse), BTN_PRIMARY, IC_FOLDER)) {
            std::string p = open_file_dialog(DLG_MODEL);
            if (!p.empty()) chat_open(p);
        }
        ImGui::SameLine();
    }
    bool can_current = A.model && A.model->spec.kind == MODEL_GPT && A.model->spec.media == MEDIA_TEXT;
    ImGui::BeginDisabled(!can_current);
    if (button(tr(S::ChatUseCurrent), BTN_SECONDARY, IC_TRAIN)) chat_use_current();
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (button(tr(S::Refresh), BTN_GHOST, IC_REFRESH)) scan_chat_models();
    ImGui::SameLine();
    if (button(tr(S::OpenFolder), BTN_GHOST, IC_FOLDER)) {
        std::error_code ec;
        std::filesystem::create_directories(upath(models_dir()), ec);
        open_folder(models_dir());
    }
    if (!c.ok && !c.msg.empty()) {
        gap(2);
        text_wrapped(strf(tr(S::Failed), c.msg.c_str()).c_str(), C::bad, F.small);
    }
    end_card();
}

// Collects finished answers and sends waiting messages; called every frame.
void chat_pump() {
    State::Chat& c = A.chat;
    chat_collect();
    if (!c.running && !c.th.joinable() && !c.queue.empty()) {
        std::string next = c.queue.front();
        c.queue.erase(c.queue.begin());
        chat_send(next);
    }
}

void page_chat() {
    State::Chat& c = A.chat;
    const char* clear_label = tr(S::ChatClear);
    const char* serve_label = tr(S::ServeApi);
    header(tr(S::NavChat), tr(S::ChatHelp), button_width(clear_label, IC_PLUS) + button_width(serve_label, IC_GLOBE) + px(8));
    ImGui::BeginDisabled(!c.model || !chat_ready() || c.path.empty());
    if (button(serve_label, BTN_SECONDARY, IC_GLOBE)) {
        std::string cli = (upath(exe_dir()) / "kor-cli").u8string();
#if defined(_WIN32)
        cli += ".exe";
#endif
        run_in_terminal(upath(c.path).parent_path().u8string(),
                        {cli, "serve", c.path, "--port", "8080", "--q-tag", c.q_tag, "--a-tag", c.a_tag});
    }
    ImGui::EndDisabled();
    tooltip(tr(S::ServeApiHelp));
    ImGui::SameLine(0, px(8));
    ImGui::BeginDisabled(c.msgs.empty() && !c.running);
    if (button(clear_label, BTN_SECONDARY, IC_PLUS)) {
        chat_stop();
        c.msgs.clear();
        c.queue.clear();
        c.reply.clear();
        c.focus = true;
    }
    ImGui::EndDisabled();
    header_end();

    chat_model_list();
    gap();

    if (!c.model) {
        empty_state(IC_CHATS, nullptr, tr(S::ChatNoModel));
        return;
    }
    if (!chat_ready()) {
        banner(tr(S::ChatNotText), C::warn, IC_INFO);
        return;
    }

    // ---- model line and settings
    std::string sub = strf(tr(S::ChatModelFmt), fmt_count((double)c.model->param_count(false)).c_str(), c.model->spec.ctx);
    std::string summary = c.name + "   \xC2\xB7   " + sub;
    if (begin_collapsible_card("chatcfg", tr(S::Settings), &c.settings_open, summary.c_str())) {
        float lw = std::min(px(220), ImGui::GetContentRegionAvail().x * 0.4f);
        field_label(tr(S::Temperature), lw);
        slider_float("##ctemp", &c.temperature, 0.f, 1.5f);
        field_label(tr(S::ChatMaxLen), lw);
        slider_int("##clen", &c.max_len, 10, 1000);
        field_label(tr(S::ChatTags), lw, tr(S::ChatTagsHelp));
        float half = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        ImGui::SetNextItemWidth(half);
        ImGui::InputText("##cq", c.q_tag, sizeof c.q_tag);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::InputText("##ca", c.a_tag, sizeof c.a_tag);
        gap(2);
        checkbox(tr(S::ChatMemory), &c.memory);
        tooltip(tr(S::ChatMemoryHelp));
    }
    end_card();
    gap();

    // ---- conversation
    float input_h = ImGui::GetFrameHeight() + px(8);
    float h = std::max(px(260), ImGui::GetContentRegionAvail().y - input_h - px(6));
    if (begin_inset("##chatlog", h)) {
        if (c.msgs.empty() && !c.running) {
            ImVec2 sz = text_size(ImGui::GetFont(), tr(S::ChatEmpty));
            ImVec2 avail = ImGui::GetContentRegionAvail();
            ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPosX() + std::max(0.f, (avail.x - sz.x) * 0.5f),
                                       ImGui::GetCursorPosY() + std::max(0.f, (avail.y - sz.y) * 0.5f)));
            text(tr(S::ChatEmpty), C::subtle);
        }
        for (size_t i = 0; i < c.msgs.size(); ++i) {
            ImGui::PushID((int)i);
            chat_bubble(c.msgs[i].text, c.msgs[i].user, false);
            ImGui::PopID();
        }
        if (c.running || c.th.joinable()) {
            std::string partial;
            {
                std::lock_guard<std::mutex> lk(c.mu);
                partial = c.reply;
            }
            ImGui::PushID("typing");
            chat_bubble(partial, false, true);
            ImGui::PopID();
            c.scroll = true;
        }
        for (size_t i = 0; i < c.queue.size(); ++i) {
            ImGui::PushID((int)(10000 + i));
            chat_bubble(c.queue[i], true, false);
            ImGui::PopID();
        }
        if (c.scroll) {
            ImGui::SetScrollHereY(1.f);
            c.scroll = false;
        }
    }
    ImGui::EndChild();
    gap(6);

    // ---- input
    bool generating = c.running || c.th.joinable();
    const char* act = tr(generating ? S::Stop : S::ChatSend);
    float bw = button_width(act, generating ? IC_STOP : IC_ARROW_UP);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - bw - ImGui::GetStyle().ItemSpacing.x);
    if (c.focus) {
        ImGui::SetKeyboardFocusHere();
        c.focus = false;
    }
    bool enter = ImGui::InputTextWithHint("##chatin", tr(S::ChatInputHint), c.input, sizeof c.input, ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    bool click = generating ? button(act, BTN_DANGER, IC_STOP) : button(act, BTN_PRIMARY, IC_ARROW_UP);
    if (generating && click) {
        c.cancel = true;  // the answer so far is kept
    } else if ((enter || click) && c.input[0]) {
        chat_send(c.input);
        c.input[0] = 0;
        c.focus = true;
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

// -------------------------------------------------------------- title bar

// Marks the last item as clickable, so the window is not dragged from it.
void title_hot() {
    ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
    A.title_hot.push_back(ImVec4(a.x, a.y, b.x, b.y));
}

bool bar_icon(const char* id, Icon ic, ImVec2 p, float s, const char* tip, const ImVec4* dot = nullptr) {
    ImGui::SetCursorScreenPos(p);
    bool c = ImGui::InvisibleButton(id, ImVec2(s, s));
    title_hot();
    bool hov = ImGui::IsItemHovered();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (hov) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        dl->AddRectFilled(p, ImVec2(p.x + s, p.y + s), col(C::surface2), px(8));
        if (tip) tooltip(tip);
    }
    icon(dl, ic, ImVec2(p.x + s * 0.5f, p.y + s * 0.5f), px(17), col(hov ? C::text : mix(C::muted, C::text, 0.45f)));
    if (dot) {
        ImVec2 dc(p.x + s * 0.72f, p.y + s * 0.28f);
        dl->AddCircleFilled(dc, px(4.5f), col(C::sidebar));
        dl->AddCircleFilled(dc, px(3), col(*dot));
    }
    return c;
}

void notifications_popup() {
    ImGui::SetNextWindowSize(ImVec2(px(360), 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(px(14), px(12)));
    if (ImGui::BeginPopup("##notif")) {
        {
            FontScope f(F.bold);
            ImGui::TextUnformatted(tr(S::Notifications));
        }
        gap(4);
        ImFont* bf = ImGui::GetFont();
        ImFont* sf = F.small ? F.small : bf;
        if (update_ready()) {
            float w = ImGui::GetContentRegionAvail().x, h = px(44);
            ImVec2 p = ImGui::GetCursorScreenPos();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), col(C::accent, 0.14f), px(8));
            dl->AddRect(p, ImVec2(p.x + w, p.y + h), col(C::accent, 0.5f), px(8));
            icon(dl, IC_SPARK, ImVec2(p.x + px(16), p.y + h * 0.5f), px(15), col(C::accent_hi));
            std::string msg = strf(tr(S::UpdateAvailable), A.update->latest.c_str());
            float bw = button_width(tr(S::UpdateDownload));
            text_ellipsis(dl, bf, ImVec2(p.x + px(32), p.y + (h - bf->FontSize) * 0.5f), w - px(44) - bw, msg.c_str(), col(C::text));
            ImGui::SetCursorScreenPos(ImVec2(p.x + w - bw - px(6), p.y + (h - ImGui::GetFrameHeight()) * 0.5f));
            if (button(tr(S::UpdateDownload), BTN_PRIMARY)) open_folder(A.update->url);
            ImGui::SetCursorScreenPos(ImVec2(p.x, p.y + h));
            ImGui::Dummy(ImVec2(0, 0));
            gap(6);
        }
        int shown = 0;
        for (auto it = A.activity.rbegin(); it != A.activity.rend() && shown < 8; ++it, ++shown) {
            int k = std::max(0, std::min((int)State::ACT_COUNT - 1, it->kind));
            const ImVec4& dot = k == State::ACT_TRAIN_STOP ? C::warn : k == State::ACT_TRAIN_DONE ? C::good : C::accent;
            float w = ImGui::GetContentRegionAvail().x, h = px(40);
            ImVec2 p = ImGui::GetCursorScreenPos();
            ImGui::Dummy(ImVec2(w, h));
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddCircleFilled(ImVec2(p.x + px(5), p.y + px(10)), px(4), col(dot));
            text_ellipsis(dl, bf, ImVec2(p.x + px(18), p.y + px(2)), w - px(18), act_text(*it).c_str(), col(C::text));
            dl->AddText(sf, sf->FontSize, ImVec2(p.x + px(18), p.y + px(4) + bf->FontSize), col(C::subtle), ago(it->t).c_str());
        }
        if (A.activity.empty() && !update_ready()) text_wrapped(tr(S::NoNotifications), C::muted, F.small);
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar();
}

void title_bar(ImVec2 pos, float w, float h, float sbw) {
    A.title_hot.clear();
    A.title_h = h;
    ImGuiWindowFlags wf = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                          ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollWithMouse;
    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize(ImVec2(w, h));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, C::sidebar);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin("##titlebar", nullptr, wf);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddLine(ImVec2(pos.x, pos.y + h - 0.5f), ImVec2(pos.x + w, pos.y + h - 0.5f), col(C::border));
    ImFont* bf = F.bold ? F.bold : ImGui::GetFont();
    ImFont* sf = F.small ? F.small : ImGui::GetFont();
    float cy = pos.y + h * 0.5f;

    // brand
    float lz = px(28), x = pos.x + px(16);
    draw_logo(dl, ImVec2(x, cy - lz * 0.5f), lz);
    x += lz + px(10);
    {
        FontScope f(F.h2 ? F.h2 : bf);
        ImFont* hf = ImGui::GetFont();
        dl->AddText(hf, hf->FontSize, ImVec2(x, cy - hf->FontSize * 0.5f), col(C::text), "KOR Studio");
        x += text_size(hf, "KOR Studio").x + px(10);
    }
    {
        const char* v = "v" KOR_VERSION;
        ImVec2 ts = text_size(sf, v);
        ImVec2 a(x, cy - ts.y * 0.5f - px(3)), b(x + ts.x + px(14), cy + ts.y * 0.5f + px(3));
        dl->AddRectFilled(a, b, col(C::surface2), (b.y - a.y) * 0.5f);
        dl->AddRect(a, b, col(C::border_hi), (b.y - a.y) * 0.5f);
        dl->AddText(sf, sf->FontSize, ImVec2(a.x + px(7), a.y + px(3)), col(C::muted), v);
        x = b.x;
    }

    // window buttons
    float right = pos.x + w;
    if (A.custom_frame) {
        float bw = px(46);
        right -= bw * 3;
        static const int cmds[3] = {WIN_MINIMIZE, WIN_MAXIMIZE, WIN_CLOSE};
        for (int i = 0; i < 3; ++i) {
            ImVec2 p(right + bw * i, pos.y);
            ImGui::SetCursorScreenPos(p);
            ImGui::PushID(i);
            bool c = ImGui::InvisibleButton("##winbtn", ImVec2(bw, h - 1));
            ImGui::PopID();
            title_hot();
            bool hov = ImGui::IsItemHovered();
            if (hov) dl->AddRectFilled(p, ImVec2(p.x + bw, p.y + h - 1), col(i == 2 ? ImVec4(0.82f, 0.20f, 0.24f, 1.f) : C::surface2));
            Icon ic = i == 0 ? IC_MINUS : i == 1 ? (A.maximized ? IC_RESTORE : IC_MAXIMIZE) : IC_X;
            icon(dl, ic, ImVec2(p.x + bw * 0.5f, cy), px(i == 2 ? 13 : 12), col(hov ? C::text : mix(C::muted, C::text, 0.5f)));
            if (c) A.window_cmd = cmds[i];
        }
        right -= px(10);
    }

    // bell, cloud, settings
    float s = px(34), sp = px(4);
    float ix = right - s * 3 - sp * 2 - px(6);
    bool unread = A.activity.size() > A.seen_activity || (update_ready() && !A.update_seen);
    ImVec2 bell_at(ix, cy - s * 0.5f);
    if (bar_icon("##bell", IC_BELL, bell_at, s, tr(S::Notifications), unread ? &C::accent : nullptr)) {
        A.seen_activity = A.activity.size();
        A.update_seen = true;
        ImGui::OpenPopup("##notif");
    }
    bool py = A.python.found();
    if (bar_icon("##cloud", IC_CLOUD, ImVec2(ix + s + sp, cy - s * 0.5f), s, tr(py ? S::CloudOk : S::CloudOff), py ? &C::good : &C::subtle)) {
        A.page = PAGE_PLUGINS;
        if (py) A.want_community = true;
    }
    if (bar_icon("##gear", IC_GEAR, ImVec2(ix + (s + sp) * 2, cy - s * 0.5f), s, tr(S::Settings))) A.want_settings = true;
    ImGui::SetNextWindowPos(ImVec2(bell_at.x + s, pos.y + h + px(4)), ImGuiCond_Always, ImVec2(1, 0));
    notifications_popup();

    // tabs, aligned with the content
    struct Tab { int page; Icon ic; S label; };
    static const Tab tabs[] = {{PAGE_HOME, IC_HOME, S::NavHome},   {PAGE_MODEL, IC_MODEL, S::NavModel},
                               {PAGE_DATA, IC_DATA, S::NavData},   {PAGE_TRAIN, IC_TRAIN, S::NavTrain},
                               {PAGE_PLUGINS, IC_PLUGIN, S::NavPlugins}, {PAGE_PROJECTS, IC_FOLDER, S::NavProjects}};
    const int nt = (int)(sizeof tabs / sizeof tabs[0]);
    ImFont* tf = ImGui::GetFont();
    ImFont* mf = F.bold ? F.bold : tf;
    float tx = std::max(x + px(24), pos.x + sbw + px(18)), th = px(36), pad = px(14), isz = px(16), tgap = px(4);
    float full = 0;
    for (int i = 0; i < nt; ++i) full += pad * 2 + isz + px(8) + text_size(mf, tr(tabs[i].label)).x + (i ? tgap : 0);
    bool icons_only = full > ix - px(16) - tx;
    for (int i = 0; i < nt; ++i) {
        const Tab& t = tabs[i];
        const char* label = tr(t.label);
        float tw = icons_only ? px(40) : pad * 2 + isz + px(8) + text_size(mf, label).x;
        if (tx + tw > ix - px(8)) break;
        ImVec2 p(tx, cy - th * 0.5f);
        ImGui::SetCursorScreenPos(p);
        ImGui::PushID(i);
        bool c = ImGui::InvisibleButton("##tab", ImVec2(tw, th));
        ImGui::PopID();
        title_hot();
        bool hov = ImGui::IsItemHovered(), sel = A.page == t.page;
        if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        if (icons_only && hov) tooltip(label);
        float k = anim(ImGui::GetItemID(), hov);
        if (sel) dl->AddRectFilled(p, ImVec2(p.x + tw, p.y + th), col(mix(C::surface3, C::accent, 0.30f)), px(9));
        else if (k > 0.01f) dl->AddRectFilled(p, ImVec2(p.x + tw, p.y + th), col(C::surface2, k), px(9));
        ImU32 fg = col(sel ? C::text : mix(C::muted, C::text, 0.4f * k + 0.5f));
        if (icons_only) {
            icon(dl, t.ic, ImVec2(p.x + tw * 0.5f, cy), isz, fg);
        } else {
            icon(dl, t.ic, ImVec2(p.x + pad + isz * 0.5f, cy), isz, fg);
            ImFont* lf = sel && F.bold ? F.bold : tf;
            dl->AddText(lf, lf->FontSize, ImVec2(p.x + pad + isz + px(8), cy - lf->FontSize * 0.5f), fg, label);
        }
        if (c) A.page = t.page;
        tx += tw + tgap;
    }
    ImGui::End();
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

    ImFont* bf = F.bold ? F.bold : ImGui::GetFont();
    ImFont* sf = F.small ? F.small : ImGui::GetFont();
    bool folded = A.sidebar_folded;

    // header: fold, "Workspace", new project
    {
        float s = px(28);
        ImVec2 q = ImGui::GetCursorScreenPos();
        float cw = ImGui::GetContentRegionAvail().x;
        auto small_btn = [&](const char* id, Icon ic, ImVec2 at, const char* tip, bool framed) {
            ImGui::SetCursorScreenPos(at);
            bool c = ImGui::InvisibleButton(id, ImVec2(s, s));
            bool hov = ImGui::IsItemHovered();
            if (hov) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                tooltip(tip);
            }
            if (hov || framed) dl->AddRectFilled(at, ImVec2(at.x + s, at.y + s), col(hov ? C::surface3 : C::surface2), px(7));
            if (framed) dl->AddRect(at, ImVec2(at.x + s, at.y + s), col(C::border_hi), px(7));
            icon(dl, ic, ImVec2(at.x + s * 0.5f, at.y + s * 0.5f), px(14), col(hov ? C::text : mix(C::muted, C::text, 0.4f)));
            return c;
        };
        if (folded) {
            if (small_btn("##unfold", IC_SIDEBAR, ImVec2(q.x + (cw - s) * 0.5f, q.y), tr(S::UnfoldSidebar), false)) A.sidebar_folded = false;
        } else {
            if (small_btn("##fold", IC_X, q, tr(S::FoldSidebar), false)) A.sidebar_folded = true;
            dl->AddText(bf, bf->FontSize, ImVec2(q.x + s + px(6), q.y + (s - bf->FontSize) * 0.5f), col(C::text), tr(S::WorkspaceTitle));
            if (small_btn("##newp", IC_PLUS, ImVec2(q.x + cw - s, q.y), tr(S::NewProject), true)) ask_new_project();
        }
        ImGui::SetCursorScreenPos(ImVec2(q.x, q.y + s));
        ImGui::Dummy(ImVec2(0, 0));
        gap(10);
    }

    TrainStats st = A.trainer.stats();
    static const S nav[] = {S::NavHome, S::NavProjects, S::NavData, S::NavModel, S::NavTrain, S::NavTest, S::NavChat, S::NavFiles, S::NavPlugins};
    static const Icon icons[] = {IC_HOME, IC_FOLDER, IC_DATA, IC_MODEL, IC_TRAIN, IC_CHAT, IC_CHATS, IC_FILES, IC_PLUGIN};
    static_assert(sizeof(nav) / sizeof(nav[0]) == PAGE_COUNT, "one label per page");
    static const int order[] = {PAGE_HOME, PAGE_PROJECTS, PAGE_MODEL, PAGE_DATA, PAGE_TRAIN, PAGE_TEST, PAGE_CHAT, PAGE_FILES, PAGE_PLUGINS};
    static_assert(sizeof(order) / sizeof(order[0]) == PAGE_COUNT, "every page in the menu");
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(px(8), px(3)));
    for (int i : order) {
        int status = 0;
        if (i == PAGE_TRAIN && busy()) status = 2;
        if (i == PAGE_PLUGINS && A.runner.running()) status = 2;
        if (i == PAGE_CHAT && A.chat.running) status = 2;
        ImGui::PushID(i);
        if (nav_item(tr(nav[i]), icons[i], A.page == i, status)) A.page = i;
        ImGui::PopID();
    }
    ImGui::PopStyleVar();

    // ---- projects
    ImFont* lf = F.small_bold ? F.small_bold : ImGui::GetFont();
    bool show_train = !folded && st.total > 0 && st.state != TS_IDLE;
    float card_h = px(12) * 2 + lf->FontSize + px(8) + px(4) + px(8) + sf->FontSize;
    float user_h = px(52), disk_h = folded ? 0.f : lf->FontSize + px(8) + px(5) + px(16);
    float block = user_h + px(10) + disk_h + (show_train ? card_h + px(12) : 0);
    if (!folded) {
        if (!A.projects_scanned) scan_projects();
        float room = ImGui::GetWindowHeight() - px(18) - block - ImGui::GetCursorPosY() - px(24);
        float rh = px(48);
        int fit = std::clamp((int)(room / rh), 0, 3);
        int n = std::min(fit, (int)A.projects.size());
        if (n > 0) {
            gap(6);
            ImVec2 q = ImGui::GetCursorScreenPos();
            float lw = ImGui::GetContentRegionAvail().x;
            dl->AddLine(ImVec2(q.x, q.y), ImVec2(q.x + lw, q.y), col(C::border));
            gap(10);
            for (int i = 0; i < n; ++i) {
                const State::ProjectInfo& pi = A.projects[i];
                ImGui::PushID(i);
                ImVec2 c = ImGui::GetCursorScreenPos();
                if (ImGui::InvisibleButton("##sproj", ImVec2(lw, rh))) project_open_from_list(pi.dir);
                bool hov = ImGui::IsItemHovered();
                if (hov) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    tooltip(pi.dir.c_str());
                }
                bool open = !A.project_dir.empty() && pi.dir == A.project_dir;
                float t = anim(ImGui::GetItemID(), hov);
                if (open || t > 0.01f)
                    dl->AddRectFilled(c, ImVec2(c.x + lw, c.y + rh), col(open ? mix(C::surface2, C::accent, 0.16f) : C::surface2, open ? 1.f : t), px(9));
                icon(dl, project_icon(pi.kind), ImVec2(c.x + px(18), c.y + rh * 0.5f), px(16), col(open ? C::accent_hi : C::muted));
                float x = c.x + px(38), tw = lw - px(38) - px(8);
                float y0 = c.y + (rh - bf->FontSize - sf->FontSize - px(2)) * 0.5f;
                text_ellipsis(dl, ImGui::GetFont(), ImVec2(x, y0), tw, pi.name.c_str(), col(C::text));
                text_ellipsis(dl, sf, ImVec2(x, y0 + bf->FontSize + px(2)), tw, project_desc(pi).c_str(), col(C::subtle));
                ImGui::PopID();
            }
        }
    }

    // ---- bottom: training card, disk usage, user
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
    if (!folded) {
        refresh_disk();
        ImVec2 q = ImGui::GetCursorScreenPos();
        float lw = ImGui::GetContentRegionAvail().x;
        dl->AddText(lf, lf->FontSize, q, col(C::muted), tr(S::DiskUsage));
        std::string v = A.disk_total_gb > 0 ? strf("%.0f GB / %.0f GB", A.disk_used_gb, A.disk_total_gb) : std::string("-");
        float vw = text_size(sf, v.c_str()).x;
        dl->AddText(sf, sf->FontSize, ImVec2(q.x + lw - vw, q.y + (lf->FontSize - sf->FontSize) * 0.5f), col(C::subtle), v.c_str());
        float by = q.y + lf->FontSize + px(8);
        float frac = A.disk_total_gb > 0 ? (float)std::clamp(A.disk_used_gb / A.disk_total_gb, 0.0, 1.0) : 0.f;
        dl->AddRectFilled(ImVec2(q.x, by), ImVec2(q.x + lw, by + px(5)), col(C::surface3), px(3));
        if (frac > 0) {
            ImVec2 a(q.x, by), b(q.x + std::max(px(5), lw * frac), by + px(5));
            int v0 = dl->VtxBuffer.Size;
            dl->AddRectFilled(a, b, col(C::accent), px(3));
            for (int k = v0; k < dl->VtxBuffer.Size; ++k) {
                float tt = (dl->VtxBuffer[k].pos.x - a.x) / std::max(1.f, b.x - a.x);
                dl->VtxBuffer[k].col = col(mix(C::accent, C::accent_hi, tt));
            }
        }
        ImGui::Dummy(ImVec2(lw, disk_h));
    }
    {
        ImVec2 q = ImGui::GetCursorScreenPos();
        float lw = ImGui::GetContentRegionAvail().x;
        float sidepad = ImGui::GetStyle().WindowPadding.x;
        dl->AddLine(ImVec2(q.x - sidepad, q.y), ImVec2(q.x + lw + sidepad, q.y), col(C::border));
        ImGui::Dummy(ImVec2(0, px(10)));
        q = ImGui::GetCursorScreenPos();
        if (ImGui::InvisibleButton("##user", ImVec2(lw, user_h - px(10)))) ImGui::OpenPopup("##usermenu");
        bool hov = ImGui::IsItemHovered();
        if (hov) ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        float rh = user_h - px(10);
        if (hov) dl->AddRectFilled(q, ImVec2(q.x + lw, q.y + rh), col(C::surface2), px(9));
        float av = px(32);
        ImVec2 ac(folded ? q.x + lw * 0.5f : q.x + px(4) + av * 0.5f, q.y + rh * 0.5f);
        dl->AddCircleFilled(ac, av * 0.5f, col(mix(C::surface3, C::accent, 0.45f)), 32);
        std::string initial = "?";
        if (!A.user.empty()) {
            unsigned char c0 = (unsigned char)A.user[0];
            size_t n = c0 < 0x80 ? 1 : c0 < 0xE0 ? 2 : c0 < 0xF0 ? 3 : 4;
            initial = A.user.substr(0, n);
            if (n == 1) initial[0] = (char)std::toupper(c0);
        }
        ImVec2 isz = text_size(bf, initial.c_str());
        dl->AddText(bf, bf->FontSize, ImVec2(ac.x - isz.x * 0.5f, ac.y - isz.y * 0.5f), col(C::text), initial.c_str());
        if (folded) {
            if (hov) tooltip(A.user.c_str());
        } else {
            float x = q.x + px(4) + av + px(10), tw = lw - (x - q.x) - px(22);
            float y0 = q.y + (rh - bf->FontSize - sf->FontSize - px(2)) * 0.5f;
            text_ellipsis(dl, bf, ImVec2(x, y0), tw, A.user.empty() ? "-" : A.user.c_str(), col(C::text));
            dl->AddText(sf, sf->FontSize, ImVec2(x, y0 + bf->FontSize + px(2)), col(C::subtle), tr(S::LocalAccount));
            icon(dl, IC_CHEVRON_DOWN, ImVec2(q.x + lw - px(10), q.y + rh * 0.5f), px(12), col(C::muted));
        }
        ImGui::SetNextWindowPos(ImVec2(q.x, q.y - px(6)), ImGuiCond_Always, ImVec2(0, 1));
        ImGui::SetNextWindowSizeConstraints(ImVec2(px(210), 0), ImVec2(px(320), FLT_MAX));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(px(10), px(10)));
        if (ImGui::BeginPopup("##usermenu")) {
            {
                FontScope fs(F.small_bold);
                text(tr(S::Language), C::subtle);
            }
            for (int l = 0; l < LANG_COUNT; ++l)
                if (ImGui::Selectable(language_name(l), l == get_language())) set_language(l);
            ImGui::Separator();
            if (ImGui::Selectable(tr(S::Settings))) A.want_settings = true;
            if (ImGui::Selectable(tr(S::Terminal))) open_terminal_here();
            ImGui::EndPopup();
        }
        ImGui::PopStyleVar();
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
        else if (key == "chat" && !value.empty()) A.chat.recent.push_back(value);
        else if (key == "project" && !value.empty()) A.recent_projects.push_back(value);
        else if (key == "dataset" && !value.empty()) A.recent_data.push_back(value);
        else if (key == "fold") A.sidebar_folded = value == "1";
        else if (key == "update_check") A.update_check = value != "0";
        else if (key == "gpu") A.use_gpu = value == "1";
    }
    set_language(lang);
    const char* shown_user = std::getenv("KOR_USER");
    A.user = shown_user && *shown_user ? shown_user : user_name();
    load_hero();
    scan_all_plugins();
    A.python.start();
    refresh_resume();
    load_activity();
    A.seen_activity = A.activity.size();
}

void shutdown() {
    A.runner.cancel();
    A.trainer.stop();  // an unfinished run saves itself here
    chat_stop();
    stop_media_gen();
    stop_sound();
    if (!A.save_settings) return;
    save_open_project();
    for (size_t i = 0; i < A.plugins.size() && i < A.plugin_args.size(); ++i)
        A.saved_args[A.plugins[i].file] = A.plugin_args[i].data();
    std::ofstream out(upath(settings_file()), std::ios::binary);
    out << "lang " << get_language() << "\n";
    for (auto& s : A.skills_on) out << "skill " << s << "\n";
    for (auto& r : A.chat.recent) out << "chat " << r << "\n";
    for (auto& r : A.recent_projects) out << "project " << r << "\n";
    for (auto& r : A.recent_data) out << "dataset " << r << "\n";
    out << "fold " << (A.sidebar_folded ? 1 : 0) << "\n";
    out << "update_check " << (A.update_check ? 1 : 0) << "\n";
    out << "gpu " << (A.use_gpu ? 1 : 0) << "\n";
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

bool chat_load(const char* path) { return chat_open(path); }

bool open_project(const char* path) {
    std::string err;
    bool ok = open_project_dir(path, err);
    A.project_ok = ok && err.empty();
    A.project_msg = err;
    return ok;
}

bool save_project_to(const char* dir) {
    std::string err, name = upath(dir).filename().u8string();
    if (!save_project(dir, name, A.project_desc, err)) {
        A.project_ok = false;
        A.project_msg = err;
        return false;
    }
    A.project_dir = upath(dir).u8string();
    A.project_name = name;
    remember_project(A.project_dir);
    A.projects_scanned = false;
    return true;
}

void chat_say(const char* text) { chat_send(text); }

void generate_media() {
    if (A.model && A.model->spec.kind == MODEL_GPT && A.model->spec.media != MEDIA_TEXT) start_media_gen();
}

void start_training() { begin_training(); }

void set_train_options(int steps, int gpu) {
    if (steps > 0) {
        A.cfg.steps = steps;
        A.cfg.probe_every = std::max(1, steps / 10);
        if (A.cfg.eval_every > 0) A.cfg.eval_every = std::max(1, std::min(A.cfg.eval_every, steps / 10));
    }
    if (gpu >= 0) A.use_gpu = gpu != 0;
}

void set_train_tab(int tab) { A.train_tab = std::max(0, std::min(4, tab)); }

void use_skills_for_session(const char* const* files, int n) {
    A.skills_on.clear();
    for (int i = 0; i < n; ++i) A.skills_on.insert(files[i]);
}

void keep_settings() { A.save_settings = false; }

void scroll_to_end() { A.scroll_end = true; }

void load_community_list() { A.want_community = true; }

void open_settings() { A.want_settings = true; }

void open_clean() {
    if (A.data.kind != DATA_TEXT) return;
    A.want_clean = true;
    A.page = PAGE_DATA;
}

void check_updates() {
    A.update_started = true;
    start_update_check();
}

void set_custom_frame(bool on, bool maximized) {
    A.custom_frame = on;
    A.maximized = maximized;
}

bool title_bar_drag_at(float x, float y) {
    if (!A.custom_frame || y < 0 || y >= A.title_h) return false;
    for (const ImVec4& r : A.title_hot)
        if (x >= r.x && x < r.z && y >= r.y && y < r.w) return false;
    return true;
}

int take_window_command() {
    int c = A.window_cmd;
    A.window_cmd = WIN_NONE;
    return c;
}

void frame() {
    if (!A.update_started) {
        A.update_started = true;  // after keep_settings(): screenshot runs stay offline
        if (A.update_check && A.save_settings) start_update_check();
    }
    pump_events();
    poll_plugin();
    chat_pump();
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    float tbh = px(52), sbw = A.sidebar_folded ? px(68) : px(232);
    title_bar(vp->WorkPos, vp->WorkSize.x, tbh, sbw);
    sidebar(ImVec2(vp->WorkPos.x, vp->WorkPos.y + tbh), sbw, vp->WorkSize.y - tbh);

    ImGuiWindowFlags wf = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                          ImGuiWindowFlags_NoBringToFrontOnFocus;
    wf &= ~ImGuiWindowFlags_NoScrollbar;
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + sbw, vp->WorkPos.y + tbh));
    ImGui::SetNextWindowSize(ImVec2(vp->WorkSize.x - sbw, vp->WorkSize.y - tbh));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(px(18), px(16)));
    ImGui::Begin("##content", nullptr, wf);
    ImGui::PopStyleVar();
    ImGui::PushID(A.page);
    switch (A.page) {
        case PAGE_HOME: page_home(); break;
        case PAGE_PROJECTS: page_projects(); break;
        case PAGE_DATA: page_data(); break;
        case PAGE_MODEL: page_model(); break;
        case PAGE_TRAIN: page_train(); break;
        case PAGE_TEST: page_test(); break;
        case PAGE_CHAT: page_chat(); break;
        case PAGE_FILES: page_files(); break;
        case PAGE_PLUGINS: page_plugins(); break;
    }
    ImGui::PopID();
    gap(4);
    shortcuts();
    new_project_modal();
    settings_modal();
    command_palette();
    if (A.page != PAGE_TEST) pump_generation();
    if (A.want_community && A.python.found() && !A.runner.running()) {
        A.want_community = false;
        run_hub(State::RUN_INDEX, {"search", "--json"});
    }
    if (A.scroll_end && ImGui::GetScrollMaxY() > 0) ImGui::SetScrollY(ImGui::GetScrollMaxY());
    ImGui::End();
}

}  // namespace app
