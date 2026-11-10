// Kor Studio - neural network layers with hand-written backward passes.
//
// Layers keep pointers to their inputs between forward() and backward()
// instead of copying them (to save memory), so the caller must keep the
// input tensors alive until backward() has run.
#pragma once
#include "tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kor {

struct Layer {
    virtual ~Layer() {}
    virtual void forward(const Tensor& x, Tensor& y, bool train) = 0;
    // Writes dL/dx into dx and accumulates parameter gradients.
    virtual void backward(const Tensor& dy, Tensor& dx) = 0;
    virtual void collect(std::vector<Param*>&) {}
};

// Fully connected layer: y = x W + b, with W stored as [in x out].
// Optionally carries a LoRA adapter: y += (x A) B * scale, with W frozen.
struct Linear : Layer {
    int in = 0, out = 0;
    Param W, b;
    bool need_dx = true;  // false for the first layer of a network

    int rank = 0;
    float lscale = 1.f;
    Param A, Bm;

    Linear(int in, int out, const std::string& name, Rng& rng, float stddev = -1.f);
    void enable_lora(int r, Rng& rng);
    void merge_lora();  // folds the adapter into W and removes it

    void forward(const Tensor& x, Tensor& y, bool train) override;
    void backward(const Tensor& dy, Tensor& dx) override;
    void collect(std::vector<Param*>& ps) override;

private:
    const Tensor* xin = nullptr;
    Tensor xa, dxa;
};

enum ActKind { ACT_RELU = 0, ACT_GELU, ACT_TANH, ACT_SIGMOID };

struct Activation : Layer {
    int kind;
    explicit Activation(int k) : kind(k) {}
    void forward(const Tensor& x, Tensor& y, bool train) override;
    void backward(const Tensor& dy, Tensor& dx) override;

private:
    const Tensor* xin = nullptr;
    const Tensor* yout = nullptr;
};

struct Dropout : Layer {
    float p;
    Rng rng;
    Dropout(float p_, uint64_t seed) : p(p_), rng(seed) {}
    void forward(const Tensor& x, Tensor& y, bool train) override;
    void backward(const Tensor& dy, Tensor& dx) override;

private:
    std::vector<uint8_t> mask;
    bool active = false;
};

struct LayerNorm : Layer {
    int dim;
    Param g, b;
    LayerNorm(int dim, const std::string& name);
    void forward(const Tensor& x, Tensor& y, bool train) override;
    void backward(const Tensor& dy, Tensor& dx) override;
    void collect(std::vector<Param*>& ps) override;

private:
    Tensor xhat;
    std::vector<float> inv;
};

// Token + learned position embedding.
struct Embedding {
    int vocab, ctx, dim;
    Param tok, pos;
    Embedding(int vocab, int ctx, int dim, Rng& rng);
    void forward(const std::vector<int>& ids, int B, int T, Tensor& y);
    void backward(const Tensor& dy);
    void collect(std::vector<Param*>& ps);

private:
    const std::vector<int>* ids_in = nullptr;
    int B = 0, T = 0;
};

// Causal multi-head self-attention over rows laid out as [B*T x C].
struct Attention {
    int C, H, hd;
    Linear qkv, proj;
    Attention(int dim, int heads, const std::string& name, Rng& rng, float proj_std);
    void forward(const Tensor& x, Tensor& y, int B, int T);
    void backward(const Tensor& dy, Tensor& dx);

private:
    int B = 0, T = 0;
    Tensor q3, ho, dho, dq3;
    std::vector<float> att;
};

// Pre-norm transformer block: x + attn(ln(x)), then h + mlp(ln(h)).
struct Block {
    LayerNorm ln1, ln2;
    Attention attn;
    Linear fc1, fc2;
    Activation act;
    Block(int dim, int heads, const std::string& name, Rng& rng, float resid_std);
    void forward(const Tensor& x, Tensor& y, int B, int T);
    void backward(const Tensor& dy, Tensor& dx);
    void collect(std::vector<Param*>& ps);
    void linears(std::vector<Linear*>& ls);

private:
    Tensor a1, a2, h, a3, a4, a5, a6, t1, t2, dh;
};

// Losses. They return the mean loss; when d is non-null they write the
// gradient w.r.t. the input multiplied by `scale`.
float cross_entropy(const Tensor& logits, const int* target, Tensor* d, float scale, int* correct);
float mse_loss(const Tensor& pred, const Tensor& y, Tensor* d, float scale);

}  // namespace kor
