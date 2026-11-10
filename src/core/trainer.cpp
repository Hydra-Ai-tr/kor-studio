#include "trainer.h"

#include "kernels.h"
#include "parallel.h"

#include <chrono>
#include <cmath>

namespace kor {

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
    else if (model->spec.kind == MODEL_GPT && data->vocab != model->vocab) err = "dataset vocabulary does not match the model";
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
        TrainEvent e;
        e.kind = EV_START;
        e.n1 = (long long)model.param_count(true);
        e.n2 = get_threads();
        e.text = K().name;
        emit(e);
    }
    Batch batch, vbatch;
    double active = 0;  // seconds spent training (pauses excluded)
    float ema = 0;
    int final_state = TS_DONE, last_step = 0;
    std::string error;

    // Early stopping follows the validation loss. The best-weights snapshot
    // follows the question tests when they run during training (most passed
    // wins; a later tie wins), otherwise - or while no test has passed yet -
    // the validation loss.
    float best_val = -1;
    int best_step = 0, stale = 0;
    bool use_probes = text && !cfg.probes.empty(), target_hit = false;
    bool probe_best = use_probes && cfg.probe_every > 0;
    std::vector<std::vector<float>> snap_w;
    int snap_step = 0, snap_pass = -1;
    auto snapshot = [&](int step) {
        snap_step = step;
        snap_w.resize(ps.size());
        for (size_t i = 0; i < ps.size(); ++i)
            if (ps[i]->trainable) snap_w[i] = ps[i]->w.v;
    };
    auto restore = [&] {
        for (size_t i = 0; i < ps.size(); ++i)
            if (ps[i]->trainable && snap_w[i].size() == ps[i]->w.v.size()) ps[i]->w.v = snap_w[i];
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

    for (int step = 1; step <= cfg.steps; ++step) {
        while (paused_ && !stop_) {
            { std::lock_guard<std::mutex> lk(stats_mu_); stats_.state = TS_PAUSED; }
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }
        if (stop_) { final_state = TS_STOPPED; break; }
        auto t0 = clock::now();
        float loss = 0, lr = 0, val_loss = -1, val_acc = -1;
        bool evaluated = false, improved = false, early = false;
        {
            std::lock_guard<std::mutex> lk(model_mu);
            model.zero_grad();
            for (int a = 0; a < accum; ++a) {
                data.sample(batch, B, T, rng, false);
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
                opt.step(ps, lr);
                if (cfg.eval_every > 0 && (step % cfg.eval_every == 0 || step == cfg.steps) && data.val_items() > 0) {
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
                if (probe_best && step % cfg.probe_every == 0 && step != cfg.steps) {
                    int pass = run_probes(step);
                    target_hit = reached_target(pass);
                    if (cfg.keep_best && pass > 0 && pass >= snap_pass) {
                        snap_pass = pass;
                        snapshot(step);
                    }
                }
            }
        }
        if (final_state == TS_ERROR) break;
        active += std::chrono::duration<double>(clock::now() - t0).count();
        ema = step == 1 ? loss : 0.9f * ema + 0.1f * loss;
        last_step = step;
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
        if (target_hit) {
            TrainEvent e;
            e.kind = EV_TARGET;
            e.step = step; e.time = active;
            { std::lock_guard<std::mutex> lk(stats_mu_); e.n1 = stats_.probe_pass; e.n2 = stats_.probe_total; }
            emit(e);
            break;
        }
        if (early) {
            TrainEvent e;
            e.kind = EV_EARLY_STOP;
            e.step = step; e.time = active; e.n1 = cfg.patience;
            emit(e);
            break;
        }
    }

    // When the question tests decided to stop, the current weights are the ones that passed them.
    if (final_state != TS_ERROR && !target_hit && last_step > 0) {
        std::lock_guard<std::mutex> lk(model_mu);
        TrainEvent e;
        e.kind = EV_RESTORED;
        e.time = active;
        bool stopped = final_state == TS_STOPPED;  // the user is waiting: skip the slow tests
        if (probe_best && !stopped) {
            int pass = run_probes(last_step);
            if (cfg.keep_best && !snap_w.empty() && snap_pass > pass) {
                restore();
                e.step = snap_step;
                e.n1 = snap_pass;
                e.n2 = (long long)cfg.probes.size();
                emit(e);
                run_probes(last_step);
            } else if (cfg.keep_best && !snap_w.empty() && snap_pass < 1 && pass == 0 && snap_step != last_step) {
                restore();  // no test ever passed: fall back to the lowest validation loss
                e.step = snap_step;
                e.val_loss = best_val;
                emit(e);
                run_probes(last_step);
            }
        } else if (!probe_best) {
            if (cfg.keep_best && !snap_w.empty() && snap_step != last_step) {
                restore();
                e.step = snap_step;
                e.val_loss = best_val;
                emit(e);
            }
            if (use_probes && !stopped) run_probes(last_step);
        }
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
