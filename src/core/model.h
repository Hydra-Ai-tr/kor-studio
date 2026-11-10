// Kor Studio - model description, construction, inference and checkpoints.
#pragma once
#include "layers.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace kor {

enum ModelKind { MODEL_MLP = 0, MODEL_GPT = 1 };
enum LayerType { L_DENSE = 0, L_RELU, L_GELU, L_TANH, L_SIGMOID, L_DROPOUT, L_LAYERNORM, L_TYPE_COUNT };

struct LayerSpec {
    int type = L_DENSE;
    int units = 64;   // L_DENSE
    float p = 0.1f;   // L_DROPOUT
};

// Everything needed to rebuild a network's structure.
struct ModelSpec {
    int kind = MODEL_MLP;
    // MLP (tabular data)
    int in_dim = 0, out_dim = 0;
    bool regression = false;
    std::vector<LayerSpec> layers;  // hidden stack; the output layer is implicit
    // GPT (character-level text)
    int vocab = 0, ctx = 64, dim = 64, heads = 4, nlayer = 2;
    // LoRA rank of adapters attached to every Linear (0 = none)
    int lora_rank = 0;

    std::string to_text() const;
    bool from_text(const std::string& s);
};

size_t estimate_params(const ModelSpec& s);
// Rough peak memory (bytes) needed to train with the given batch size.
size_t estimate_train_bytes(const ModelSpec& s, int batch);

// One mini-batch. MLP uses x (+ tgt or y); GPT uses ids + tgt with B, T.
struct Batch {
    Tensor x, y;
    std::vector<int> ids, tgt;
    int B = 0, T = 0;
};

std::filesystem::path upath(const std::string& utf8);

class Model {
public:
    ModelSpec spec;
    // GPT: token strings. MLP classifier: class labels.
    std::vector<std::string> vocab;
    // MLP: feature names and standardization (last entry = target for regression).
    std::vector<std::string> feat_names;
    std::vector<float> norm_mean, norm_std;

    explicit Model(const ModelSpec& s, uint64_t seed = 1234);

    // Forward pass, loss, and (if train) backward pass that ACCUMULATES
    // gradients scaled by `scale`. Returns the mean loss.
    float step(const Batch& b, bool train, float scale, int* correct = nullptr);
    void zero_grad();
    std::vector<Param*> params();
    size_t param_count(bool trainable_only = false);

    void enable_lora(int rank);
    void merge_lora();

    // MLP inference on raw (unnormalized) features: class probabilities, or
    // a single de-normalized value for regression.
    std::vector<float> predict(const std::vector<float>& features);
    // GPT inference: samples the next token id given previous ids.
    int next_token(const std::vector<int>& ids, float temperature, Rng& rng);
    std::vector<int> encode(const std::string& text) const;

    bool save(const std::string& path, bool half_precision, std::string* err);
    static std::shared_ptr<Model> load(const std::string& path, std::string* err);

private:
    void forward(const Batch& b, bool train);
    const Tensor& output() const;

    Rng rng;
    std::vector<Linear*> all_linears;
    // MLP
    std::vector<std::unique_ptr<Layer>> layers;
    std::vector<Tensor> acts;
    // GPT
    std::unique_ptr<Embedding> emb;
    std::vector<std::unique_ptr<Block>> blocks;
    std::unique_ptr<LayerNorm> lnf;
    std::unique_ptr<Linear> head;
    std::vector<Tensor> xs;
    Tensor xf, logits;
    // scratch
    Tensor dout, d1, d2;
    Batch infer;
};

uint16_t float_to_half(float f);
float half_to_float(uint16_t h);
// Splits UTF-8 text into code points (as strings); drops '\r'.
std::vector<std::string> utf8_chars(const std::string& s);

}  // namespace kor
