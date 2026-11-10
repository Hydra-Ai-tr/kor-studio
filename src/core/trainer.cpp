#include "trainer.h"

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
        stats_.total = cfg.steps;
        if (!err.empty()) { stats_.state = TS_ERROR; stats_.error = err; return false; }
        stats_.state = TS_RUNNING;
    }
    model_ = model; data_ = data; cfg_ = cfg;
    stop_ = false; paused_ = false; running_ = true;
    th_ = std::thread([this] { run(); });
    return true;
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
    Rng rng(cfg.seed * 7919 + 13), val_rng(999);
    Optimizer opt(cfg.optimizer, cfg.weight_decay);
    std::vector<Param*> ps;
    {
        std::lock_guard<std::mutex> lk(model_mu);
        ps = model.params();
    }
    Batch batch, vbatch;
    double active = 0;  // seconds spent training (pauses excluded)
    float ema = 0;
    int final_state = TS_DONE;
    std::string error;

    for (int step = 1; step <= cfg.steps; ++step) {
        while (paused_ && !stop_) {
            { std::lock_guard<std::mutex> lk(stats_mu_); stats_.state = TS_PAUSED; }
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }
        if (stop_) { final_state = TS_STOPPED; break; }
        auto t0 = clock::now();
        float loss = 0, lr = 0, val_loss = -1, val_acc = -1;
        bool evaluated = false;
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
                }
            }
        }
        if (final_state == TS_ERROR) break;
        active += std::chrono::duration<double>(clock::now() - t0).count();
        ema = step == 1 ? loss : 0.9f * ema + 0.1f * loss;

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
        }
    }
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_.state = final_state;
        stats_.error = error;
        stats_.eta = 0;
    }
    running_ = false;
}

}  // namespace kor
