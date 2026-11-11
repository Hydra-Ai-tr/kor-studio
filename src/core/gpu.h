// Kor Studio - experimental GPU matrix products through OpenCL.
//
// OpenCL is loaded at run time (OpenCL.dll, libOpenCL.so, the macOS framework),
// so the programs start and work without it. When turned on, text models run
// their whole forward and backward pass on the GPU (gpu_gpt_step) while the
// optimizer stays on the CPU; other models send their attention and large matrix
// products. Work that does not fit in GPU memory runs on the CPU, and any other
// GPU error switches back to the CPU for the rest of the run.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace kor {

struct Param;

struct GpuInfo {
    bool found = false;
    std::string name, vendor, platform;
    double mem_gb = 0;
    std::string error;  // why no GPU is usable
};

// Looks for an OpenCL GPU once and remembers the answer.
const GpuInfo& gpu_probe();

// Turns GPU matrix products on (checks the results against the CPU and
// measures from which size the GPU is faster) or off. Returns false and stays
// on the CPU when the GPU cannot help; `msg` says what happened either way.
bool gpu_use(bool on, std::string* msg = nullptr);
bool gpu_active();

// C += op(A) * op(B) on the GPU, with the argument order of Kernels::mm (variant 0),
// mm_ta (1) and mm_tb (2). For tests and benchmarks.
bool gpu_gemm(int variant, const float* A, const float* B, float* C, int x, int y, int z, std::string* err);

// Causal self-attention on the GPU (see Kernels::attn_fwd / attn_bwd): forward
// writes `res` = out, backward writes `res` = dqkv. For tests and benchmarks.
bool gpu_attention(bool backward, const float* qkv, const float* dout, float* res, int B, int T, int H, int hd, std::string* err);
// Largest relative difference of GPU attention (forward and backward) from a
// plain double-precision version on random data; -1 when the GPU cannot run it.
double gpu_attention_check(int B, int T, int H, int hd, std::string* err);

// A text model (MODEL_GPT without LoRA) for gpu_gpt_step; params in the order
// of Model::params(): token and position embedding, 12 per block, final norm, head.
struct GpuGpt {
    int vocab = 0, ctx = 0, dim = 0, heads = 0, nlayer = 0;
    std::vector<Param*> params;
};

// The whole forward pass (and with `train` the backward pass) of a text model on
// the GPU while gpu_use() is on; the activations never leave the card. Like
// Model::step it returns the mean loss and ADDS scale * gradient to the
// parameters' gradients. Dropout masks come from `seed`. Returns false (and
// changes nothing) when the GPU cannot do it, e.g. the model does not fit.
bool gpu_gpt_step(const GpuGpt& m, const int* ids, const int* tgt, const float* w, int B, int T, bool train, float dropout,
                  uint64_t seed, float scale, float* loss, int* correct, std::string* err = nullptr);

}  // namespace kor
