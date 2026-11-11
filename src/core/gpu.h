// Kor Studio - experimental GPU matrix products through OpenCL.
//
// OpenCL is loaded at run time (OpenCL.dll, libOpenCL.so, the macOS framework),
// so the programs start and work without it. When turned on, text models run
// their whole forward and backward pass on the GPU (gpu_gpt_step), and in a
// resident run the optimizer too; other models send their attention and large
// matrix products. Work that does not fit in GPU memory runs on the CPU, and any other
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
// parameters' gradients. Dropout masks come from `seed`. A batch that does not
// fit in GPU memory at once runs in parts. Returns false (and changes nothing on
// the host) when the GPU cannot do it, e.g. the model does not fit.
//
// During a resident run of the same parameters the gradients stay on the card;
// if the step fails there, the run ends and its weights and optimizer moments
// are back on the host, so the caller redoes the whole step on the CPU.
bool gpu_gpt_step(const GpuGpt& m, const int* ids, const int* tgt, const float* w, int B, int T, bool train, float dropout,
                  uint64_t seed, float scale, float* loss, int* correct, std::string* err = nullptr);

// At most `rows` sequences per pass of gpu_gpt_step (0 = as many as fit); for tests.
void gpu_limit_pass_rows(int rows);

// Resident training: weights, gradients and optimizer moments of `m` live on the
// card between steps, so a step copies only the tokens. `moments` is 2 for AdamW
// and 1 for SGD with momentum; the host moments (when allocated) are the start.
// Begin fails when the model, its gradients, moments and one sequence of B x T
// do not fit. The host weights are stale until gpu_resident_pull().
bool gpu_resident_begin(const GpuGpt& m, int moments, int B, int T, std::string* err = nullptr);
bool gpu_resident_active();
bool gpu_resident_zero_grad();
// Optimizer::step on the card after clipping the global gradient norm to `clip`
// (0 = no clipping); t is the step count after this step. False with a
// non-finite norm (weights unchanged, run still on) or on a GPU error (run ended,
// state back on the host).
bool gpu_resident_update(bool adam, float lr, float weight_decay, float clip, long t, float* grad_norm);
// Copies the weights (and moments) to the host / the host weights to the card.
bool gpu_resident_pull(bool moments);
bool gpu_resident_push();
// Ends the run, with `pull` after copying its state to the host.
bool gpu_resident_end(bool pull);

}  // namespace kor
