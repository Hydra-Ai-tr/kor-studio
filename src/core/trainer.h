// Kor Studio - optimizer and background training loop.
#pragma once
#include "data.h"
#include "model.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace kor {

enum OptKind { OPT_ADAMW = 0, OPT_SGD = 1 };

struct TrainConfig {
    int steps = 1000;
    int batch = 32;
    int accum = 1;        // gradient accumulation: effective batch = batch * accum
    float lr = 3e-3f;
    float weight_decay = 0.01f;
    int warmup = 30;
    float clip = 1.0f;    // global gradient-norm clip (0 = off)
    int threads = 0;      // 0 = keep current
    int eval_every = 50;
    int eval_batches = 8;
    int optimizer = OPT_ADAMW;
    uint64_t seed = 1;
    // Keep a copy of the best weights and return to them when training ends
    // (protects against overfitting). "Best" means the most question tests
    // passed when tests run during training, else the lowest validation loss.
    bool keep_best = true;
    // Stop when validation loss has not improved for this many evaluations
    // in a row (0 = never stop early).
    int patience = 0;
    // Question tests for text models, run every `probe_every` steps (0 = only
    // at the end) and once more when training finishes.
    std::vector<Probe> probes;
    int probe_every = 0;
    // Stop as soon as at least this fraction of the question tests pass
    // (0 = off, 1 = all of them). The weights at that moment are kept.
    float probe_target = 0;
};

enum TrainEventKind { EV_START = 0, EV_EVAL, EV_PROBES, EV_EARLY_STOP, EV_TARGET, EV_RESTORED, EV_DONE, EV_STOPPED, EV_ERROR };

// One entry of the training log. Fields are filled depending on `kind`:
//   EV_START      n1 = trainable parameters, n2 = threads, text = kernel name
//   EV_EVAL       loss, val_loss, val_acc (-1 if n/a), n1 = 1 when this is the best so far
//   EV_PROBES     n1 = passed, n2 = total, text = "question -> answer" example
//   EV_EARLY_STOP n1 = patience
//   EV_TARGET     n1 = passed, n2 = total (the question-test target was reached)
//   EV_RESTORED   step = step of the restored weights; n2 > 0: n1/n2 question tests
//                 passed with them, otherwise val_loss = their validation loss
//   EV_ERROR      text = message
struct TrainEvent {
    int kind = 0;
    int step = 0;
    double time = 0;  // seconds of training so far
    float loss = 0, val_loss = -1, val_acc = -1;
    long long n1 = 0, n2 = 0;
    std::string text;
};

class Optimizer {
public:
    Optimizer(int kind, float weight_decay) : kind_(kind), wd_(weight_decay) {}
    void step(const std::vector<Param*>& ps, float lr);

private:
    int kind_;
    float wd_;
    long t_ = 0;
};

// Scales gradients so their global L2 norm is at most max_norm; returns the norm.
float clip_gradients(const std::vector<Param*>& ps, float max_norm);

enum TrainState { TS_IDLE = 0, TS_RUNNING, TS_PAUSED, TS_DONE, TS_STOPPED, TS_ERROR };

struct TrainStats {
    int state = TS_IDLE;
    int step = 0, total = 0;
    float loss = 0, val_loss = -1, val_acc = -1, lr = 0;
    double items_per_sec = 0, elapsed = 0, eta = 0;
    std::vector<float> train_hist;          // smoothed loss per step
    std::vector<float> val_hist;            // validation loss per evaluation
    std::vector<int> val_steps;
    float best_val = -1;
    int best_step = 0;
    int probe_pass = -1, probe_total = 0;    // -1 until the tests have run
    std::vector<ProbeResult> probe_results;  // latest run, same order as cfg.probes
    std::vector<int> probe_steps, probe_hist;  // passed tests per run
    std::string error;
};

class Trainer {
public:
    ~Trainer();
    // Starts training in a background thread. The dataset must outlive the run.
    bool start(std::shared_ptr<Model> model, const Dataset* data, const TrainConfig& cfg);
    void pause(bool p) { paused_ = p; }
    void stop();                 // requests stop and joins
    bool running() const { return running_; }
    bool paused() const { return paused_; }
    TrainStats stats();
    // Moves log entries produced since the last call into `out`.
    void drain_events(std::vector<TrainEvent>& out);

    // Held by the trainer for the duration of each step; lock it to use the
    // model from another thread (inference, saving) while training.
    std::mutex model_mu;

private:
    void run();
    void emit(const TrainEvent& e);
    std::vector<TrainEvent> events_;
    std::shared_ptr<Model> model_;
    const Dataset* data_ = nullptr;
    TrainConfig cfg_;
    std::thread th_;
    std::atomic<bool> running_{false}, paused_{false}, stop_{false};
    std::mutex stats_mu_;
    TrainStats stats_;
};

}  // namespace kor
