// Kor Studio - hardware detection and automatic configuration.
#pragma once
#include "data.h"
#include "trainer.h"

#include <string>
#include <vector>

namespace kor {

struct SysInfo {
    std::string cpu = "Unknown CPU";
    int logical = 1;
    double ram_total_gb = 0, ram_free_gb = 0;
    bool avx2 = false, avx512 = false;
    std::vector<std::string> gpus;  // informational: training runs on the CPU
};

bool cpu_has_avx2();
SysInfo detect_system();
// Threads that give good throughput while keeping the machine responsive.
int recommended_threads(const SysInfo& s);

struct AutoPlan {
    ModelSpec spec;
    TrainConfig cfg;
    int tier = 0;            // 0 nano .. 4 large on the CPU, up to 9 (about 300M parameters) on a GPU
    size_t params = 0;
    double est_ram_mb = 0;
    double budget_mb = 0;
    double est_gpu_mb = 0;   // GPU memory of the planned model (tiers above 4)
};

// Picks a model size and training settings that fit this machine and dataset.
// gpu_gb > 0 plans text models for training on a GPU with that much memory
// (weights, gradients and optimizer there): larger models when the data has
// enough text for them.
AutoPlan autotune(const SysInfo& s, const Dataset& d, double gpu_gb = 0);

// GPU memory (bytes) to train a text model resident on the card: parameters,
// gradients, two optimizer moments and the activations of one sequence (with
// `recompute` only the block inputs, see Model::recompute).
double estimate_gpu_train_bytes(const ModelSpec& s, bool recompute = false);

}  // namespace kor
