// Kor Studio - numeric kernels with runtime CPU dispatch (AVX2/FMA or baseline).
#pragma once

namespace kor {

struct Kernels {
    const char* name;
    // C[m x n] += A[m x k] * B[k x n]
    void (*mm)(const float* A, const float* B, float* C, int m, int k, int n);
    // C[k x n] += A[m x k]^T * B[m x n]
    void (*mm_ta)(const float* A, const float* B, float* C, int m, int k, int n);
    // C[m x k] += A[m x n] * B[k x n]^T
    void (*mm_tb)(const float* A, const float* B, float* C, int m, int n, int k);
    float (*dot)(const float* a, const float* b, int n);
    void (*axpy)(float a, const float* x, float* y, int n);  // y += a * x
    // Optional causal self-attention on an accelerator (null on the CPU). qkv holds
    // [B*T x 3C] rows (queries, keys, values; C = H * hd). attn_fwd writes
    // out [B*T x C]; attn_bwd turns dout [B*T x C] into dqkv [B*T x 3C].
    // Both overwrite their result and return false when the caller must use the CPU.
    bool (*attn_fwd)(const float* qkv, float* out, int B, int T, int H, int hd);
    bool (*attn_bwd)(const float* qkv, const float* dout, float* dqkv, int B, int T, int H, int hd);
};

// The kernels in use: the fastest CPU set, or the GPU set while gpu_use() is on.
const Kernels& K();
// The fastest kernel set supported by this CPU.
const Kernels& cpu_kernels();
// Replaces the kernels returned by K() (nullptr = back to the CPU set).
void set_kernels(const Kernels* k);

}  // namespace kor
