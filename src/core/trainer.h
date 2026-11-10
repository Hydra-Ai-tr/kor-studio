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
    int eval_batches = 4;
    int optimizer = OPT_ADAMW;
    uint64_t seed = 1;
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

    // Held by the trainer for the duration of each step; lock it to use the
    // model from another thread (inference, saving) while training.
    std::mutex model_mu;

private:
    void run();
    std::shared_ptr<Model> model_;
    const Dataset* data_ = nullptr;
    TrainConfig cfg_;
    std::thread th_;
    std::atomic<bool> running_{false}, paused_{false}, stop_{false};
    std::mutex stats_mu_;
    TrainStats stats_;
};

}  // namespace kor
