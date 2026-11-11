#include "trainer.h"

#include "kernels.h"
#include "parallel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace kor {

// --------------------------------------------------------------- checkpoints

namespace {

const char kStateMagic[4] = {'K', 'O', 'R', 'S'};
const uint32_t kStateVersion = 1;

struct Out {
    std::ofstream o;
    template <class T> void put(const T& v) { o.write((const char*)&v, sizeof(T)); }
    void str(const std::string& s) { put((uint32_t)s.size()); o.write(s.data(), (std::streamsize)s.size()); }
    template <class T> void vec(const std::vector<T>& v) {
        put((uint64_t)v.size());
        if (!v.empty()) o.write((const char*)v.data(), (std::streamsize)(v.size() * sizeof(T)));
    }
};

struct In {
    std::ifstream i;
    bool ok = true;
    template <class T> T get() {
        T v{};
        if (ok && !i.read((char*)&v, sizeof(T))) ok = false;
        return v;
    }
    std::string str() {
        uint32_t n = get<uint32_t>();
        if (!ok || n > (1u << 26)) { ok = false; return {}; }
        std::string s(n, '\0');
        if (n && !i.read(&s[0], n)) ok = false;
        return s;
    }
    template <class T> std::vector<T> vec() {
        uint64_t n = get<uint64_t>();
        if (!ok || n > (1ull << 31) / sizeof(T)) { ok = false; return {}; }
        std::vector<T> v((size_t)n);
        if (n && !i.read((char*)v.data(), (std::streamsize)(n * sizeof(T)))) ok = false;
        return v;
    }
};

void put_config(Out& w, const TrainConfig& c) {
    w.put((int32_t)c.steps); w.put((int32_t)c.batch); w.put((int32_t)c.accum);
    w.put(c.lr); w.put(c.weight_decay); w.put((int32_t)c.warmup); w.put(c.clip);
    w.put((int32_t)c.eval_every); w.put((int32_t)c.eval_batches); w.put((int32_t)c.optimizer);
    w.put((uint64_t)c.seed);
    w.put(c.align); w.put(c.prompt_weight); w.put(c.dropout); w.put(c.ema);
    w.put((uint8_t)(c.keep_best ? 1 : 0)); w.put((int32_t)c.patience);
    w.put((int32_t)c.probe_every); w.put(c.probe_target);
    w.put(c.checkpoint_every);
}

void get_config(In& r, TrainConfig& c) {
    c.steps = r.get<int32_t>(); c.batch = r.get<int32_t>(); c.accum = r.get<int32_t>();
    c.lr = r.get<float>(); c.weight_decay = r.get<float>(); c.warmup = r.get<int32_t>(); c.clip = r.get<float>();
    c.eval_every = r.get<int32_t>(); c.eval_batches = r.get<int32_t>(); c.optimizer = r.get<int32_t>();
    c.seed = r.get<uint64_t>();
    c.align = r.get<float>(); c.prompt_weight = r.get<float>(); c.dropout = r.get<float>(); c.ema = r.get<float>();
    c.keep_best = r.get<uint8_t>() != 0; c.patience = r.get<int32_t>();
    c.probe_every = r.get<int32_t>(); c.probe_target = r.get<float>();
    c.checkpoint_every = r.get<double>();
}

// Progress of a run besides the model weights.
struct RunState {
    int step = 0;
    double active = 0;
    float ema_loss = 0, best_val = -1, last_val = -1, last_acc = -1;
    int best_step = 0, stale = 0, snap_step = 0, snap_pass = -1;
    long long opt_t = 0;
    float lr_factor = 1, lr_fixed = 0;
    uint64_t rng = 1;
    TrainConfig cfg;
    std::string note;
    std::vector<std::string> names;
    std::vector<std::vector<float>> w, m, s, snap, ema;
    std::vector<float> train_hist, val_hist;
    std::vector<int> val_steps, probe_steps, probe_hist;
};

// Reads the header (everything before the weights) or the whole state from one file.
bool read_state_file(const std::string& file, RunState& st, bool full, std::string* err) {
    In r;
    r.i.open(upath(file), std::ios::binary);
    if (!r.i) { if (err) *err = "no saved training"; return false; }
    char magic[4] = {};
    r.i.read(magic, 4);
    if (!r.i || std::memcmp(magic, kStateMagic, 4) != 0 || r.get<uint32_t>() != kStateVersion) {
        if (err) *err = "the saved training file is not valid";
        return false;
    }
    st.step = r.get<int32_t>();
    st.active = r.get<double>();
    st.ema_loss = r.get<float>(); st.best_val = r.get<float>(); st.last_val = r.get<float>(); st.last_acc = r.get<float>();
    st.best_step = r.get<int32_t>(); st.stale = r.get<int32_t>(); st.snap_step = r.get<int32_t>(); st.snap_pass = r.get<int32_t>();
    st.opt_t = r.get<int64_t>();
    st.lr_factor = r.get<float>(); st.lr_fixed = r.get<float>();
    st.rng = r.get<uint64_t>();
    get_config(r, st.cfg);
    st.note = r.str();
    if (full && r.ok) {
        uint32_t n = r.get<uint32_t>();
        for (uint32_t k = 0; k < n && r.ok && k < 100000; ++k) {
            st.names.push_back(r.str());
            st.w.push_back(r.vec<float>());
            st.m.push_back(r.vec<float>());
            st.s.push_back(r.vec<float>());
            st.snap.push_back(r.vec<float>());
            st.ema.push_back(r.vec<float>());
        }
        st.train_hist = r.vec<float>();
        st.val_hist = r.vec<float>();
        st.val_steps = r.vec<int>();
        st.probe_steps = r.vec<int>();
        st.probe_hist = r.vec<int>();
        char end[4] = {};
        r.i.read(end, 4);
        if (!r.i || std::memcmp(end, kStateMagic, 4) != 0) r.ok = false;  // cut short
    }
    if (!r.ok || st.step < 0) {
        if (err) *err = "the saved training file is damaged";
        return false;
    }
    return true;
}

// The latest state, or the previous one when the latest is damaged (e.g. the
// power went off while it was written).
bool read_state(const std::string& path, RunState& st, bool full, std::string* err) {
    std::string e1;
    if (read_state_file(path + ".state", st, full, &e1)) return true;
    st = RunState();
    if (read_state_file(path + ".state.bak", st, full, nullptr)) return true;
    if (err) *err = e1;
    return false;
}

// Makes sure the file's data is on the disk, not only in the system cache.
void flush_to_disk(const std::string& file) {
#ifdef _WIN32
    FILE* f = _wfopen(upath(file).c_str(), L"r+b");
    if (f) { std::fflush(f); _commit(_fileno(f)); std::fclose(f); }
#else
    FILE* f = std::fopen(file.c_str(), "r+b");
    if (f) { fsync(fileno(f)); std::fclose(f); }
#endif
}

// Moves `tmp` over `path`. With `backup`, the old file is kept as `<path>.bak`.
bool replace_file(const std::string& tmp, const std::string& path, bool backup, std::string* err) {
    std::error_code ec;
    if (backup && std::filesystem::exists(upath(path), ec)) {
        std::filesystem::remove(upath(path + ".bak"), ec);
        std::filesystem::rename(upath(path), upath(path + ".bak"), ec);
    }
    ec.clear();
    std::filesystem::rename(upath(tmp), upath(path), ec);
    if (ec) {
        std::filesystem::remove(upath(path), ec);
        std::filesystem::rename(upath(tmp), upath(path), ec);
    }
    if (ec) { if (err) *err = ec.message(); return false; }
    return true;
}

}  // namespace

bool read_checkpoint(const std::string& path, CheckpointInfo& info, std::string* err) {
    RunState st;
    if (!read_state(path, st, false, err)) return false;
    std::error_code ec;
    if (!std::filesystem::exists(upath(path), ec)) {
        if (err) *err = "the saved model file is missing";
        return false;
    }
    info.step = st.step;
    info.elapsed = st.active;
    info.best_val = st.best_val;
    info.cfg = st.cfg;
    info.note = st.note;
    return true;
}

bool has_checkpoint(const std::string& path) {
    std::error_code ec;
    return !path.empty() && std::filesystem::exists(upath(path), ec) &&
           (std::filesystem::exists(upath(path + ".state"), ec) || std::filesystem::exists(upath(path + ".state.bak"), ec));
}

void remove_checkpoint(const std::string& path) {
    std::error_code ec;
    for (const char* ext : {".state", ".state.bak", ".state.tmp", ".tmp", ""})
        std::filesystem::remove(upath(path + ext), ec);
}

void Optimizer::step(const std::vector<Param*>& ps, float lr) {
    ++t_;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float c1 = 1.f - std::pow(b1, (float)t_), c2 = 1.f - std::pow(b2, (float)t_);
    for (Param* p : ps) {
        if (!p->trainable) continue;
        size_t n = p->w.size();
        float* w = p->w.data();
        const float* g = p->g.data();
        float decay = p->decay ? wd_ : 0.f;
        if (p->m.size() != n) p->m.resize(p->w.rows, p->w.cols);
        float* m = p->m.data();
        if (kind_ == OPT_SGD) {
            for (size_t i = 0; i < n; ++i) {
                m[i] = 0.9f * m[i] + g[i];
                w[i] -= lr * (m[i] + decay * w[i]);
            }
            continue;
        }
        if (p->s.size() != n) p->s.resize(p->w.rows, p->w.cols);
        float* s = p->s.data();
        for (size_t i = 0; i < n; ++i) {
            m[i] = b1 * m[i] + (1.f - b1) * g[i];
            s[i] = b2 * s[i] + (1.f - b2) * g[i] * g[i];
            float mh = m[i] / c1, sh = s[i] / c2;
            w[i] -= lr * (mh / (std::sqrt(sh) + eps) + decay * w[i]);
        }
    }
}

float clip_gradients(const std::vector<Param*>& ps, float max_norm) {
    double sq = 0;
    for (Param* p : ps)
        if (p->trainable)
            for (float g : p->g.v) sq += (double)g * g;
    float norm = (float)std::sqrt(sq);
    if (max_norm > 0 && norm > max_norm) {
        float k = max_norm / norm;
        for (Param* p : ps)
            if (p->trainable)
                for (float& g : p->g.v) g *= k;
    }
    return norm;
}

Trainer::~Trainer() { stop(); }

void Trainer::stop() {
    stop_ = true;
    if (th_.joinable()) th_.join();
    stop_ = false;
    paused_ = false;
}

TrainStats Trainer::stats() {
    std::lock_guard<std::mutex> lk(stats_mu_);
    return stats_;
}

bool Trainer::start(std::shared_ptr<Model> model, const Dataset* data, const TrainConfig& cfg) {
    if (running_) return false;
    if (th_.joinable()) th_.join();
    std::string err;
    if (!model || !data || data->kind == DATA_NONE) err = "no model or data";
    else if (model->spec.kind == MODEL_GPT && (data->kind != DATA_TEXT || data->ids.size() < 16)) err = "text model needs a text dataset";
    else if (model->spec.kind == MODEL_GPT && data->media != model->spec.media) err = "the model and the data are of different kinds (text, images, sound)";
    else if (model->spec.kind == MODEL_GPT && data->vocab != model->vocab) err = "dataset vocabulary does not match the model";
    else if (model->spec.kind == MODEL_GPT && data->ids.size() < (size_t)model->spec.ctx + 2) err = "too little data for the context length";
    else if (model->spec.kind == MODEL_MLP && (data->kind != DATA_TABLE || data->n_feat != model->spec.in_dim)) err = "table columns do not match the model";
    else if (model->spec.kind == MODEL_MLP && !model->spec.regression && (int)data->labels.size() != model->spec.out_dim) err = "class count does not match the model";
    else if (model->param_count(true) == 0) err = "model has no trainable parameters";
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_ = TrainStats();
        events_.clear();
        stats_.total = cfg.steps;
        if (!err.empty()) { stats_.state = TS_ERROR; stats_.error = err; return false; }
        stats_.state = TS_RUNNING;
    }
    model_ = model; data_ = data; cfg_ = cfg;
    stop_ = false; paused_ = false; running_ = true;
    th_ = std::thread([this] { run(); });
    return true;
}

void Trainer::emit(const TrainEvent& e) {
    std::lock_guard<std::mutex> lk(stats_mu_);
    events_.push_back(e);
}

void Trainer::drain_events(std::vector<TrainEvent>& out) {
    std::lock_guard<std::mutex> lk(stats_mu_);
    for (auto& e : events_) out.push_back(std::move(e));
    events_.clear();
}

// --------------------------------------------------------------- hooks

class RunControl final : public TrainControl {
public:
    RunControl(Trainer& t, Model& m, const Dataset& d, const TrainConfig& c) : tr_(t), m_(m), d_(d), cfg_(c) {}

    const Model& model() const override { return m_; }
    const Dataset& data() const override { return d_; }
    const TrainConfig& config() const override { return cfg_; }

    void set_lr_factor(float factor) override {
        if (!std::isfinite(factor) || factor <= 0) return;
        lr_factor = std::max(1e-4f, std::min(100.f, factor));
        lr_fixed = 0;
        TrainEvent e = event(EV_HOOK_LR);
        e.loss = lr_factor;
        tr_.emit(e);
    }
    void set_lr(float lr) override {
        if (!std::isfinite(lr)) return;
        lr_fixed = lr > 0 ? std::min(lr, 10.f) : 0;
        TrainEvent e = event(EV_HOOK_LR);
        e.loss = lr_fixed > 0 ? lr_fixed : lr_factor;
        e.n1 = lr_fixed > 0 ? 1 : 0;
        tr_.emit(e);
    }
    void stop(const std::string& reason) override {
        if (stop_requested) return;
        stop_requested = true;
        TrainEvent e = event(EV_HOOK_STOP);
        e.text = reason;
        tr_.emit(e);
    }
    void save_best() override { best_request = 1; }
    bool restore_best() override {
        best_request = 0;
        return has_best;
    }
    size_t add_text(const std::string& text) override {
        if (m_.spec.kind != MODEL_GPT || text.empty()) return 0;
        std::vector<int> ids = m_.encode(text);
        if (ids.empty()) return 0;
        if (!added.empty() && newline >= 0 && added.back() != newline) added.push_back(newline);
        added.insert(added.end(), ids.begin(), ids.end());
        added_layout = layout_dialogue(added, q_tag, a_tag, newline);
        TrainEvent e = event(EV_HOOK_DATA);
        e.n1 = (long long)ids.size();
        e.n2 = (long long)added.size();
        tr_.emit(e);
        return ids.size();
    }
    void clear_added() override {
        added.clear();
        added_layout = TextLayout();
        TrainEvent e = event(EV_HOOK_DATA);
        tr_.emit(e);
    }
    void set_mix(float share) override { mix = std::isfinite(share) ? std::min(0.95f, share) : -1.f; }
    void set_sampling(float align, float question_weight) override {
        if (m_.spec.kind != MODEL_GPT) return;
        if (std::isfinite(align) && align >= 0) opts.align = std::min(1.f, align);
        if (std::isfinite(question_weight) && question_weight >= 0) opts.prompt_weight = std::min(1.f, question_weight);
        TrainEvent e = event(EV_HOOK_SAMPLING);
        e.loss = opts.align;
        e.val_loss = opts.prompt_weight;
        tr_.emit(e);
    }
    void set_dropout(float p) override {
        if (m_.spec.kind != MODEL_GPT || !std::isfinite(p)) return;
        dropout = std::max(0.f, std::min(0.9f, p));
        m_.set_dropout(dropout);
        TrainEvent e = event(EV_HOOK_DROPOUT);
        e.loss = dropout;
        tr_.emit(e);
    }
    void log(const std::string& text, int tone) override {
        TrainEvent e = event(EV_HOOK_LOG);
        e.text = text;
        e.n1 = tone;
        tr_.emit(e);
    }
    void metric(const std::string& name, float value) override {
        if (name.empty() || !std::isfinite(value)) return;
        std::string full = source + ": " + name;
        TrainEvent e = event(EV_HOOK_METRIC);
        e.text = name;
        e.loss = value;
        std::lock_guard<std::mutex> lk(tr_.stats_mu_);
        auto& ms = tr_.stats_.metrics;
        auto it = std::find_if(ms.begin(), ms.end(), [&](const TrainStats::Metric& m) { return m.name == full; });
        if (it == ms.end()) {
            ms.push_back({full, {}, {}});
            it = ms.end() - 1;
        }
        it->steps.push_back(step);
        it->values.push_back(value);
        tr_.events_.push_back(e);
    }
    std::string generate(const std::string& prompt, int length, float temperature, bool one_line) override {
        if (m_.spec.kind != MODEL_GPT) return "";
        std::lock_guard<std::mutex> lk(tr_.model_mu);
        std::vector<int> ids = m_.encode(prompt);
        if (ids.empty()) ids.push_back(rng.below(std::max(1, m_.spec.vocab)));
        std::string out;
        length = std::max(1, std::min(length, 4000));
        for (int t : m_.generate(ids, length, std::max(0.f, temperature), rng, one_line ? newline : -1)) {
            if (one_line && t == newline) break;
            out += m_.vocab[t];
        }
        return out;
    }
    ProbeResult ask(const Probe& p) override {
        std::lock_guard<std::mutex> lk(tr_.model_mu);
        return run_probe(m_, p);
    }

    // Mixes rows of the added text into a batch drawn from the dataset.
    void mix_into(Batch& b, Rng& r, const SampleOpts& opts) const {
        int T = b.T;
        if (added.size() < (size_t)T + 2) return;
        float share = mix >= 0 ? mix : std::min(0.5f, (float)added.size() / (float)(added.size() + d_.n_train));
        for (int i = 0; i < b.B; ++i) {
            if (r.uniform() >= share) continue;
            size_t s = pick_window(added_layout, 0, added.size(), T, opts.align, r);
            fill_window(b, i, added, s, added_layout, opts.prompt_weight);
        }
    }

    TrainEvent event(int kind) const {
        TrainEvent e;
        e.kind = kind;
        e.step = step;
        e.time = time;
        e.source = source;
        return e;
    }

    std::string source;  // the hook being called
    int step = 0;
    double time = 0;
    float lr_factor = 1, lr_fixed = 0;
    bool stop_requested = false, has_best = false;
    int best_request = -1;  // 1 = save the current weights, 0 = restore the best ones
    std::vector<int> added;
    TextLayout added_layout;
    std::vector<int> q_tag, a_tag;
    SampleOpts opts;
    float dropout = 0;
    float mix = -1;
    int newline = -1;

private:
    Trainer& tr_;
    Model& m_;
    const Dataset& d_;
    const TrainConfig& cfg_;
    Rng rng{0xC0FFEE};
};

// ------------------------------------------------------------ main loop

void Trainer::run() {
    using clock = std::chrono::steady_clock;
    const TrainConfig cfg = cfg_;
    if (cfg.threads > 0) set_threads(cfg.threads);
    Model& model = *model_;
    const Dataset& data = *data_;
    bool text = model.spec.kind == MODEL_GPT;
    int T = text ? std::min(model.spec.ctx, (int)data.ids.size() - 2) : 1;
    int B = std::max(1, cfg.batch), accum = std::max(1, cfg.accum);
    Rng rng(cfg.seed * 7919 + 13);
    Optimizer opt(cfg.optimizer, cfg.weight_decay);
    std::vector<Param*> ps;
    {
        std::lock_guard<std::mutex> lk(model_mu);
        ps = model.params();
        model.set_dropout(text ? cfg.dropout : 0.f);
        TrainEvent e;
        e.kind = EV_START;
        e.n1 = (long long)model.param_count(true);
        e.n2 = get_threads();
        e.text = K().name;
        emit(e);
    }
    Batch batch, vbatch;
    double active = 0;  // seconds spent training (pauses excluded)
    float ema = 0, lr = 0;
    int final_state = TS_DONE, last_step = 0;
    std::string error, end_state = "done";

    // Early stopping follows the validation loss. The best-weights snapshot
    // follows the question tests when they run during training (most passed
    // wins; a later tie wins), otherwise - or while no test has passed yet -
    // the validation loss. Hooks can save or restore it at any time.
    float best_val = -1, last_val = -1, last_acc = -1;
    int best_step = 0, stale = 0;
    bool use_probes = text && !cfg.probes.empty(), target_hit = false;
    bool probe_best = use_probes && cfg.probe_every > 0;
    std::vector<std::vector<float>> snap_w;
    int snap_step = 0, snap_pass = -1;
    std::vector<ProbeResult> last_results;
    auto snapshot = [&](int step) {
        snap_step = step;
        snap_w.resize(ps.size());
        for (size_t i = 0; i < ps.size(); ++i)
            if (ps[i]->trainable) snap_w[i] = ps[i]->w.v;
    };
    // Averaged weights. While `ema_in` is set they are swapped into the model
    // (evaluation, tests, snapshots) and the raw weights wait in ema_w.
    bool use_ema = cfg.ema > 0 && cfg.ema < 1;
    bool ema_in = false;
    std::vector<std::vector<float>> ema_w;
    if (use_ema) {
        ema_w.resize(ps.size());
        for (size_t i = 0; i < ps.size(); ++i)
            if (ps[i]->trainable) ema_w[i] = ps[i]->w.v;
    }
    auto ema_update = [&](int step) {
        float d = std::min(cfg.ema, (1.f + step) / (10.f + step));
        for (size_t i = 0; i < ps.size(); ++i) {
            if (!ps[i]->trainable) continue;
            float* e = ema_w[i].data();
            const float* w = ps[i]->w.data();
            size_t n = ema_w[i].size();
            for (size_t k = 0; k < n; ++k) e[k] = d * e[k] + (1.f - d) * w[k];
        }
    };
    auto ema_swap = [&] {
        if (!use_ema) return;
        for (size_t i = 0; i < ps.size(); ++i)
            if (ps[i]->trainable) std::swap(ps[i]->w.v, ema_w[i]);
        ema_in = !ema_in;
    };
    auto restore = [&] {
        for (size_t i = 0; i < ps.size(); ++i)
            if (ps[i]->trainable && snap_w[i].size() == ps[i]->w.v.size()) {
                ps[i]->w.v = snap_w[i];
                if (use_ema) ema_w[i] = snap_w[i];
            }
    };

    // Runs the question tests and returns how many passed; the caller must hold model_mu.
    auto run_probes = [&](int step) {
        std::vector<ProbeResult> results;
        int pass = 0, example = -1;
        for (size_t i = 0; i < cfg.probes.size(); ++i) {
            results.push_back(run_probe(model, cfg.probes[i]));
            if (results.back().pass) ++pass;
            else if (example < 0) example = (int)i;  // show a failing one if there is any
        }
        if (example < 0) example = step % (int)cfg.probes.size();
        TrainEvent e;
        e.kind = EV_PROBES;
        e.step = step;
        e.time = active;
        e.n1 = pass;
        e.n2 = (long long)cfg.probes.size();
        const Probe& p = cfg.probes[example];
        e.text = (p.question.empty() ? p.prompt.substr(0, p.prompt.find('\n')) : p.question) + " -> " + results[example].answer;
        last_results = results;
        std::lock_guard<std::mutex> lk(stats_mu_);
        events_.push_back(e);
        stats_.probe_pass = pass;
        stats_.probe_total = (int)cfg.probes.size();
        stats_.probe_results = std::move(results);
        stats_.probe_steps.push_back(step);
        stats_.probe_hist.push_back(pass);
        return pass;
    };
    auto reached_target = [&](int pass) {
        return cfg.probe_target > 0 && pass >= (int)std::ceil(cfg.probe_target * cfg.probes.size() - 1e-4);
    };

    // ---- hooks
    RunControl ctl(*this, model, data, cfg);
    for (size_t i = 0; i < model.vocab.size(); ++i)
        if (model.vocab[i] == "\n") ctl.newline = (int)i;
    int tag_newline = -1;
    if (text && !data.tag_ids(ctl.q_tag, ctl.a_tag, tag_newline)) {
        ctl.q_tag.clear();
        ctl.a_tag.clear();
    }
    SampleOpts& sopts = ctl.opts;
    sopts.align = text ? std::max(0.f, std::min(1.f, cfg.align)) : 0.f;
    sopts.prompt_weight = text ? std::max(0.f, std::min(1.f, cfg.prompt_weight)) : 1.f;
    ctl.dropout = text ? cfg.dropout : 0.f;
    // ---- checkpoints
    int start_step = 0;
    if (cfg.resume) {
        RunState st;
        std::string err;
        bool ok = read_state(cfg.checkpoint, st, true, &err);
        if (ok && st.names.size() != ps.size()) { ok = false; err = "the saved training belongs to a different model"; }
        for (size_t i = 0; ok && i < ps.size(); ++i) {
            const Param& p = *ps[i];
            size_t n = p.w.v.size();
            auto fits = [&](const std::vector<float>& v) { return v.empty() || v.size() == n; };
            if (st.names[i] != p.name || st.w[i].size() != n || !fits(st.m[i]) || !fits(st.s[i]) || !fits(st.snap[i]) || !fits(st.ema[i])) {
                ok = false;
                err = "the saved training belongs to a different model";
            }
        }
        if (ok && st.step >= cfg.steps) { ok = false; err = "the saved training is already finished"; }
        if (!ok) {
            std::lock_guard<std::mutex> lk(stats_mu_);
            stats_.state = TS_ERROR;
            stats_.error = err;
            TrainEvent e;
            e.kind = EV_ERROR;
            e.text = err;
            events_.push_back(e);
            running_ = false;
            return;
        }
        {
            std::lock_guard<std::mutex> lk(model_mu);
            for (size_t i = 0; i < ps.size(); ++i) {
                Param& p = *ps[i];
                p.w.v = st.w[i];
                if (!p.trainable) continue;
                if (!st.m[i].empty()) { p.m.resize(p.w.rows, p.w.cols); p.m.v = st.m[i]; }
                if (!st.s[i].empty()) { p.s.resize(p.w.rows, p.w.cols); p.s.v = st.s[i]; }
                if (use_ema) ema_w[i] = st.ema[i].empty() ? p.w.v : st.ema[i];
            }
            if (!st.snap.empty() && std::any_of(st.snap.begin(), st.snap.end(), [](const std::vector<float>& v) { return !v.empty(); })) {
                snap_w = st.snap;
                snap_step = st.snap_step;
                snap_pass = st.snap_pass;
            }
        }
        start_step = st.step;
        active = st.active;
        ema = st.ema_loss;
        best_val = st.best_val; best_step = st.best_step; stale = st.stale;
        last_val = st.last_val; last_acc = st.last_acc;
        opt.set_steps_taken((long)st.opt_t);
        rng.s = st.rng ? st.rng : 1;
        ctl.lr_factor = st.lr_factor;
        ctl.lr_fixed = st.lr_fixed;
        last_step = start_step;
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_.step = start_step;
        stats_.loss = ema;
        stats_.elapsed = active;
        stats_.train_hist = st.train_hist;
        stats_.val_hist = st.val_hist;
        stats_.val_steps = st.val_steps;
        stats_.probe_steps = st.probe_steps;
        stats_.probe_hist = st.probe_hist;
        stats_.best_val = best_val;
        stats_.best_step = best_step;
        stats_.val_loss = last_val;
        stats_.val_acc = last_acc;
        stats_.lr_factor = ctl.lr_factor;
        stats_.lr_fixed = ctl.lr_fixed;
        stats_.resumed_from = start_step;
        stats_.saved_step = start_step;
        TrainEvent e;
        e.kind = EV_RESUMED;
        e.step = start_step;
        e.time = active;
        events_.push_back(e);
    } else if (!cfg.checkpoint.empty()) {
        remove_checkpoint(cfg.checkpoint);  // an older run there is replaced by this one
    }
    bool model_saved = cfg.resume;
    auto last_save = clock::now();
    // Writes the checkpoint; call between steps (the raw weights must be in the model).
    auto save_checkpoint = [&](int step) {
        if (cfg.checkpoint.empty() || step <= 0) return;
        last_save = clock::now();
        std::string err;
        bool ok = true;
        std::vector<float> th, vh;
        std::vector<int> vs, pst, phi;
        {
            std::lock_guard<std::mutex> lk(stats_mu_);
            th = stats_.train_hist; vh = stats_.val_hist; vs = stats_.val_steps;
            pst = stats_.probe_steps; phi = stats_.probe_hist;
        }
        {
            std::lock_guard<std::mutex> lk(model_mu);
            if (!model_saved) {
                ok = model.save(cfg.checkpoint + ".tmp", false, &err);
                if (ok) flush_to_disk(cfg.checkpoint + ".tmp");
                ok = ok && replace_file(cfg.checkpoint + ".tmp", cfg.checkpoint, false, &err);
                model_saved = ok;
            }
            if (ok) {
                std::string tmp = cfg.checkpoint + ".state.tmp";
                Out w;
                w.o.open(upath(tmp), std::ios::binary | std::ios::trunc);
                if (!w.o) { ok = false; err = "cannot write the checkpoint file"; }
                else {
                    w.o.write(kStateMagic, 4);
                    w.put(kStateVersion);
                    w.put((int32_t)step);
                    w.put(active);
                    w.put(ema); w.put(best_val); w.put(last_val); w.put(last_acc);
                    w.put((int32_t)best_step); w.put((int32_t)stale); w.put((int32_t)snap_step); w.put((int32_t)snap_pass);
                    w.put((int64_t)opt.steps_taken());
                    w.put(ctl.lr_factor); w.put(ctl.lr_fixed);
                    w.put((uint64_t)rng.s);
                    put_config(w, cfg);
                    w.str(cfg.checkpoint_note);
                    w.put((uint32_t)ps.size());
                    static const std::vector<float> none;
                    for (size_t i = 0; i < ps.size(); ++i) {
                        const Param& p = *ps[i];
                        w.str(p.name);
                        w.vec(p.w.v);
                        w.vec(p.trainable ? p.m.v : none);
                        w.vec(p.trainable ? p.s.v : none);
                        w.vec(i < snap_w.size() ? snap_w[i] : none);
                        w.vec(use_ema && i < ema_w.size() ? ema_w[i] : none);
                    }
                    w.vec(th); w.vec(vh); w.vec(vs); w.vec(pst); w.vec(phi);
                    w.o.write(kStateMagic, 4);
                    w.o.close();
                    if (!w.o) { ok = false; err = "writing the checkpoint failed (is the disk full?)"; }
                }
                if (ok) {
                    flush_to_disk(tmp);
                    ok = replace_file(tmp, cfg.checkpoint + ".state", true, &err);
                }
                if (!ok) {
                    std::error_code ec;
                    std::filesystem::remove(upath(tmp), ec);
                }
            }
        }
        std::lock_guard<std::mutex> lk(stats_mu_);
        if (ok) { stats_.saved_step = step; stats_.save_error.clear(); }
        else if (stats_.save_error.empty()) {
            stats_.save_error = err;
            TrainEvent e;
            e.kind = EV_CHECKPOINT;
            e.step = step;
            e.time = active;
            e.text = err;
            events_.push_back(e);
        } else stats_.save_error = err;
    };

    std::vector<std::shared_ptr<TrainHook>> hooks;
    for (auto& h : cfg.hooks) {
        if (!h) continue;
        ctl.source = h->name();
        std::string err;
        if (h->open(ctl, &err)) hooks.push_back(h);
        else ctl.log(err.empty() ? "could not start" : err, TONE_BAD);
    }
    auto info_at = [&](int kind, int step) {
        HookInfo in;
        in.kind = kind;
        in.step = step;
        in.total = cfg.steps;
        in.elapsed = active;
        in.loss = ema;
        in.lr = lr;
        in.val_loss = last_val;
        in.val_acc = last_acc;
        in.best_val = best_val;
        in.best_step = best_step;
        in.tests = &cfg.probes;
        in.results = &last_results;
        return in;
    };
    // Calls every hook that wants this event, then applies weight requests.
    auto dispatch = [&](const HookInfo& in) {
        if (hooks.empty()) return;
        for (auto& h : hooks) {
            if (!h->alive() || !h->wants(in.kind)) continue;
            if (in.kind == HOOK_STEP && (h->step_every() <= 0 || in.step % h->step_every() != 0)) continue;
            ctl.source = h->name();
            ctl.step = in.step;
            ctl.time = active;
            ctl.has_best = !snap_w.empty();
            h->on_event(in, ctl);
            if (ctl.best_request >= 0) {
                std::lock_guard<std::mutex> lk(model_mu);
                TrainEvent e = ctl.event(EV_HOOK_BEST);
                if (ctl.best_request == 1) {
                    ema_swap();
                    snapshot(in.step);
                    ema_swap();
                    e.n1 = 1;
                    e.n2 = in.step;
                    emit(e);
                } else if (!snap_w.empty()) {
                    restore();
                    e.n1 = 0;
                    e.n2 = snap_step;
                    emit(e);
                }
                ctl.best_request = -1;
            }
        }
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_.lr_factor = ctl.lr_factor;
        stats_.lr_fixed = ctl.lr_fixed;
        stats_.added_chars = ctl.added.size();
    };
    dispatch(info_at(HOOK_START, 0));

    for (int step = start_step + 1; step <= cfg.steps && !ctl.stop_requested; ++step) {
        bool saved_on_pause = false;
        while (paused_ && !stop_) {
            { std::lock_guard<std::mutex> lk(stats_mu_); stats_.state = TS_PAUSED; }
            if (!saved_on_pause && !cfg.checkpoint.empty() && step - 1 > start_step) {
                save_checkpoint(step - 1);
                saved_on_pause = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }
        if (stop_) { final_state = TS_STOPPED; end_state = "stopped"; break; }
        auto t0 = clock::now();
        float loss = 0, val_loss = -1, val_acc = -1;
        bool evaluated = false, improved = false, early = false, probed = false;
        int probe_pass = 0;
        {
            std::lock_guard<std::mutex> lk(model_mu);
            model.zero_grad();
            for (int a = 0; a < accum; ++a) {
                data.sample(batch, B, T, rng, false, sopts);
                if (text) ctl.mix_into(batch, rng, sopts);
                loss += model.step(batch, true, 1.f / accum) / accum;
            }
            if (!std::isfinite(loss)) {
                final_state = TS_ERROR;
                error = "loss became NaN/Inf - try a lower learning rate";
            } else {
                clip_gradients(ps, cfg.clip);
                float progress = (float)(step - 1) / std::max(1, cfg.steps - 1);
                lr = step <= cfg.warmup ? cfg.lr * step / std::max(1, cfg.warmup)
                                        : cfg.lr * (0.1f + 0.45f * (1.f + std::cos(3.14159265f * progress)));
                lr = ctl.lr_fixed > 0 ? ctl.lr_fixed : lr * ctl.lr_factor;
                opt.step(ps, lr);
                if (use_ema) ema_update(step);
                bool eval_now = cfg.eval_every > 0 && (step % cfg.eval_every == 0 || step == cfg.steps) && data.val_items() > 0;
                bool probe_now = probe_best && step % cfg.probe_every == 0 && step != cfg.steps;
                if (eval_now || probe_now) ema_swap();
                if (eval_now) {
                    // The same validation batches every time, so results are comparable.
                    Rng val_rng(999);
                    double vl = 0;
                    int ok = 0, seen = 0, nb = std::max(1, cfg.eval_batches);
                    for (int e = 0; e < nb; ++e) {
                        data.sample(vbatch, B, T, val_rng, true);
                        int c = 0;
                        vl += model.step(vbatch, false, 1.f, &c);
                        ok += c; seen += B * T;
                    }
                    val_loss = (float)(vl / nb);
                    if (!(model.spec.kind == MODEL_MLP && model.spec.regression)) val_acc = (float)ok / seen;
                    evaluated = true;
                    if (best_val < 0 || val_loss < best_val) {
                        improved = true;
                        best_val = val_loss;
                        best_step = step;
                        stale = 0;
                        if (cfg.keep_best && (!probe_best || snap_pass < 1)) snapshot(step);
                    } else if (cfg.patience > 0 && ++stale >= cfg.patience) {
                        early = true;
                    }
                }
                if (probe_now) {
                    probe_pass = run_probes(step);
                    probed = true;
                    target_hit = reached_target(probe_pass);
                    if (cfg.keep_best && probe_pass > 0 && probe_pass >= snap_pass) {
                        snap_pass = probe_pass;
                        snapshot(step);
                    }
                }
                if (ema_in) ema_swap();
            }
        }
        if (final_state == TS_ERROR) break;
        active += std::chrono::duration<double>(clock::now() - t0).count();
        ema = step == 1 ? loss : 0.9f * ema + 0.1f * loss;
        last_step = step;
        if (evaluated) { last_val = val_loss; last_acc = val_acc; }
        {
            std::lock_guard<std::mutex> lk(stats_mu_);
            stats_.state = TS_RUNNING;
            stats_.step = step;
            stats_.loss = ema;
            stats_.lr = lr;
            stats_.elapsed = active;
            stats_.items_per_sec = (double)step * B * accum * T / std::max(1e-6, active);
            stats_.eta = active / step * (cfg.steps - step);
            stats_.train_hist.push_back(ema);
            if (evaluated) {
                stats_.val_loss = val_loss;
                stats_.val_acc = val_acc;
                stats_.val_hist.push_back(val_loss);
                stats_.val_steps.push_back(step);
                stats_.best_val = best_val;
                stats_.best_step = best_step;
                TrainEvent e;
                e.kind = EV_EVAL;
                e.step = step; e.time = active;
                e.loss = ema; e.val_loss = val_loss; e.val_acc = val_acc;
                e.n1 = improved ? 1 : 0;
                events_.push_back(e);
            }
        }
        if (!hooks.empty()) {
            dispatch(info_at(HOOK_STEP, step));
            if (evaluated) {
                HookInfo in = info_at(HOOK_EVAL, step);
                in.improved = improved;
                dispatch(in);
            }
            if (probed) {
                HookInfo in = info_at(HOOK_TESTS, step);
                in.passed = probe_pass;
                dispatch(in);
            }
        }
        if (target_hit) {
            TrainEvent e;
            e.kind = EV_TARGET;
            e.step = step; e.time = active;
            { std::lock_guard<std::mutex> lk(stats_mu_); e.n1 = stats_.probe_pass; e.n2 = stats_.probe_total; }
            emit(e);
            end_state = "target";
            break;
        }
        if (early) {
            TrainEvent e;
            e.kind = EV_EARLY_STOP;
            e.step = step; e.time = active; e.n1 = cfg.patience;
            emit(e);
            end_state = "early_stop";
            break;
        }
        if (!cfg.checkpoint.empty() && step < cfg.steps &&
            std::chrono::duration<double>(clock::now() - last_save).count() >= std::max(5.0, cfg.checkpoint_every))
            save_checkpoint(step);
    }
    if (ctl.stop_requested && final_state == TS_DONE) end_state = "hook_stop";
    // A stopped or failed run stays resumable; a finished one needs no checkpoint any more.
    if (!cfg.checkpoint.empty()) {
        if (final_state == TS_STOPPED) {
            if (last_step > start_step) save_checkpoint(last_step);
        } else if (final_state == TS_DONE) {
            remove_checkpoint(cfg.checkpoint);
        }
    }
    if (use_ema && final_state != TS_ERROR) {
        std::lock_guard<std::mutex> lk(model_mu);
        ema_swap();  // the averaged weights become the model
    }

    // When the question tests decided to stop, the current weights are the ones that passed them.
    int final_pass = -1;
    if (final_state != TS_ERROR && !target_hit && last_step > 0) {
        std::lock_guard<std::mutex> lk(model_mu);
        TrainEvent e;
        e.kind = EV_RESTORED;
        e.time = active;
        bool stopped = final_state == TS_STOPPED;  // the user is waiting: skip the slow tests
        if (probe_best && !stopped) {
            int pass = run_probes(last_step);
            final_pass = pass;
            if (cfg.keep_best && !snap_w.empty() && snap_pass > pass) {
                restore();
                e.step = snap_step;
                e.n1 = snap_pass;
                e.n2 = (long long)cfg.probes.size();
                emit(e);
                final_pass = run_probes(last_step);
            } else if (cfg.keep_best && !snap_w.empty() && snap_pass < 1 && pass == 0 && snap_step != last_step) {
                restore();  // no test ever passed: fall back to the lowest validation loss
                e.step = snap_step;
                e.val_loss = best_val;
                emit(e);
                final_pass = run_probes(last_step);
            }
        } else if (!probe_best) {
            if (cfg.keep_best && !snap_w.empty() && snap_step != last_step) {
                restore();
                e.step = snap_step;
                e.val_loss = best_val;
                emit(e);
            }
            if (use_probes && !stopped) final_pass = run_probes(last_step);
        }
    }
    if (!hooks.empty()) {
        if (final_pass >= 0) {
            HookInfo in = info_at(HOOK_TESTS, last_step);
            in.passed = final_pass;
            dispatch(in);
        }
        HookInfo in = info_at(HOOK_END, last_step);
        in.state = final_state == TS_ERROR ? "error" : end_state;
        dispatch(in);
        for (auto& h : hooks) h->close();
    }
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_.state = final_state;
        stats_.error = error;
        stats_.eta = 0;
        TrainEvent e;
        e.kind = final_state == TS_ERROR ? EV_ERROR : final_state == TS_STOPPED ? EV_STOPPED : EV_DONE;
        e.step = last_step; e.time = active; e.loss = ema; e.val_loss = best_val; e.text = error;
        events_.push_back(e);
    }
    running_ = false;
}

}  // namespace kor
