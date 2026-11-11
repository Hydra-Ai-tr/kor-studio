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

class TrainHook;

struct TrainConfig {
    int steps = 1000;
    int batch = 32;
    int accum = 1;        // gradient accumulation: effective batch = batch * accum
    float lr = 3e-3f;
    float weight_decay = 0.01f;
    int warmup = 30;
    float clip = 1.0f;    // global gradient-norm clip (0 = off)
    int threads = 0;      // 0 = keep current
    bool gpu = false;     // experimental: large matrix products on an OpenCL GPU (see gpu.h)
    int eval_every = 50;
    int eval_batches = 8;
    int optimizer = OPT_ADAMW;
    uint64_t seed = 1;
    // Text batches (see SampleOpts): share of rows that start at an example,
    // and the loss weight of question text in dialogue data.
    float align = 0;
    float prompt_weight = 1;
    // Dropout of the transformer blocks (text models), against overfitting.
    float dropout = 0;
    // Exponential moving average of the weights (e.g. 0.999; 0 = off). When on,
    // evaluation, question tests, the best-weights snapshot and the final
    // model use the averaged weights, which are usually smoother.
    float ema = 0;
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
    // Code that watches the run and steers it (Python skills, see TrainHook).
    std::vector<std::shared_ptr<TrainHook>> hooks;
    // Crash safety. With a path, the run is saved there every
    // `checkpoint_every` seconds and when it is stopped or fails: the model as
    // a .kor file and the progress (step, optimizer moments, best and averaged
    // weights, curves) next to it in `<path>.state`. Both files are written
    // to a temporary name first, so a power cut never leaves a broken file. A
    // run that finishes removes them. `resume` continues a saved run: the
    // caller loads the model from `checkpoint` first and passes the settings
    // from read_checkpoint(). `checkpoint_note` is stored as is (the caller's
    // own details, e.g. where the data came from).
    std::string checkpoint;
    double checkpoint_every = 60;
    bool resume = false;
    std::string checkpoint_note;
};

// What a checkpoint holds besides the model (see TrainConfig::checkpoint).
struct CheckpointInfo {
    int step = 0;
    double elapsed = 0;
    float best_val = -1;
    TrainConfig cfg;   // settings of the run (without probes and hooks)
    std::string note;
};
bool read_checkpoint(const std::string& path, CheckpointInfo& info, std::string* err);
bool has_checkpoint(const std::string& path);
void remove_checkpoint(const std::string& path);

enum TrainEventKind {
    EV_START = 0, EV_EVAL, EV_PROBES, EV_EARLY_STOP, EV_TARGET, EV_RESTORED, EV_DONE, EV_STOPPED, EV_ERROR,
    EV_HOOK_LOG, EV_HOOK_METRIC, EV_HOOK_STOP, EV_HOOK_LR, EV_HOOK_DATA, EV_HOOK_BEST,
    EV_HOOK_SAMPLING, EV_HOOK_DROPOUT, EV_RESUMED, EV_CHECKPOINT
};
enum HookTone { TONE_INFO = 0, TONE_GOOD, TONE_WARN, TONE_BAD, TONE_MUTED };

// One entry of the training log. Fields are filled depending on `kind`:
//   EV_START      n1 = trainable parameters, n2 = threads, text = kernel name
//   EV_EVAL       loss, val_loss, val_acc (-1 if n/a), n1 = 1 when this is the best so far
//   EV_PROBES     n1 = passed, n2 = total, text = "question -> answer" example
//   EV_EARLY_STOP n1 = patience
//   EV_TARGET     n1 = passed, n2 = total (the question-test target was reached)
//   EV_RESTORED   step = step of the restored weights; n2 > 0: n1/n2 question tests
//                 passed with them, otherwise val_loss = their validation loss
//   EV_ERROR      text = message
//   EV_RESUMED    step = step the run continues from
//   EV_CHECKPOINT step = saved step; text = error message if saving failed
//   EV_HOOK_*     source = hook name
//     LOG         text = message, n1 = HookTone
//     METRIC      text = metric name, loss = value
//     STOP        text = reason
//     LR          loss = new learning-rate factor (or fixed rate when n1 = 1)
//     DATA        n1 = characters added, n2 = all added characters
//     BEST        n1 = 1 saved the current weights as the best, 0 restored the best ones
//     SAMPLING    loss = share of aligned rows, val_loss = question weight
//     DROPOUT     loss = dropout rate
struct TrainEvent {
    int kind = 0;
    int step = 0;
    double time = 0;  // seconds of training so far
    float loss = 0, val_loss = -1, val_acc = -1;
    long long n1 = 0, n2 = 0;
    std::string text, source;
};

// ---- hooks
//
// A hook (for example a Python skill) is told what happens during training and
// can change the run: the learning rate, the training text, when to stop and
// which weights to keep. Hooks run on the training thread between steps; the
// trainer waits for them, so a hook should return quickly.

enum HookKind { HOOK_START = 0, HOOK_STEP, HOOK_EVAL, HOOK_TESTS, HOOK_END };

struct HookInfo {
    int kind = HOOK_START;
    int step = 0, total = 0;
    double elapsed = 0;
    float loss = 0, lr = 0;                   // smoothed training loss, current learning rate
    float val_loss = -1, val_acc = -1;        // HOOK_EVAL (and the latest values otherwise)
    float best_val = -1;
    int best_step = 0;
    bool improved = false;                    // HOOK_EVAL: new best validation loss
    int passed = 0;                           // HOOK_TESTS
    const std::vector<Probe>* tests = nullptr;
    const std::vector<ProbeResult>* results = nullptr;
    std::string state;                        // HOOK_END: "done", "stopped", "early_stop", "target", "hook_stop", "error"
};

class Model;
struct Dataset;

// What a hook may do. Valid only during TrainHook::on_event().
class TrainControl {
public:
    virtual ~TrainControl() = default;
    virtual const Model& model() const = 0;
    virtual const Dataset& data() const = 0;
    virtual const TrainConfig& config() const = 0;
    // Learning rate = schedule * factor (default 1), or a fixed rate when set (> 0).
    virtual void set_lr_factor(float factor) = 0;
    virtual void set_lr(float lr) = 0;
    virtual void stop(const std::string& reason) = 0;
    virtual void save_best() = 0;     // keep the current weights as the best ones
    virtual bool restore_best() = 0;  // go back to the best weights kept so far
    // Text models: adds training text (characters the model does not know are
    // dropped). Returns the number of characters added.
    virtual size_t add_text(const std::string& text) = 0;
    virtual void clear_added() = 0;
    // Share of training rows taken from the added text (< 0 = automatic).
    virtual void set_mix(float share) = 0;
    // Text batches: share of rows that start at an example and the loss weight
    // of question text (see SampleOpts). Negative values keep the current one.
    virtual void set_sampling(float align, float question_weight) = 0;
    virtual void set_dropout(float p) = 0;
    virtual void log(const std::string& text, int tone) = 0;
    virtual void metric(const std::string& name, float value) = 0;
    // Text models: continues `prompt` with the current weights.
    virtual std::string generate(const std::string& prompt, int length, float temperature, bool one_line) = 0;
    virtual ProbeResult ask(const Probe& p) = 0;
};

class TrainHook {
public:
    virtual ~TrainHook() = default;
    virtual std::string name() const = 0;
    // Called first on the training thread; false drops the hook (the reason goes to `err`).
    virtual bool open(TrainControl& ctl, std::string* err) = 0;
    virtual bool wants(int hook_kind) const = 0;
    virtual int step_every() const { return 0; }  // HOOK_STEP period, 0 = never
    virtual void on_event(const HookInfo& info, TrainControl& ctl) = 0;
    virtual void close() {}
    // False once the hook has failed; it is not called again.
    virtual bool alive() const { return true; }
};

class Optimizer {
public:
    Optimizer(int kind, float weight_decay) : kind_(kind), wd_(weight_decay) {}
    void step(const std::vector<Param*>& ps, float lr);
    long steps_taken() const { return t_; }
    void set_steps_taken(long t) { t_ = t; }

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
    // Values reported by hooks ("source: name"), in the order they first appeared.
    struct Metric {
        std::string name;
        std::vector<int> steps;
        std::vector<float> values;
    };
    std::vector<Metric> metrics;
    float lr_factor = 1, lr_fixed = 0;
    size_t added_chars = 0;
    int resumed_from = 0;    // step a resumed run continued from
    int saved_step = 0;      // step of the latest checkpoint (0 = none yet)
    std::string save_error;  // why the latest checkpoint failed
};

class Trainer {
public:
    ~Trainer();
    // Starts training in a background thread. The dataset must outlive the run.
    bool start(std::shared_ptr<Model> model, const Dataset* data, const TrainConfig& cfg);
    void pause(bool p) { paused_ = p; }
    void stop();                 // requests stop and joins
    void request_stop() { stop_ = true; }  // asks the run to stop (and save) without waiting
    bool running() const { return running_; }
    bool paused() const { return paused_; }
    TrainStats stats();
    // Moves log entries produced since the last call into `out`.
    void drain_events(std::vector<TrainEvent>& out);

    // Held by the trainer for the duration of each step; lock it to use the
    // model from another thread (inference, saving) while training.
    std::mutex model_mu;

private:
    friend class RunControl;
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
