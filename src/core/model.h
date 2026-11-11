// Kor Studio - model description, construction, inference and checkpoints.
#pragma once
#include "layers.h"

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace kor {

enum ModelKind { MODEL_MLP = 0, MODEL_GPT = 1 };
// What the tokens of a sequence model (MODEL_GPT) stand for.
enum MediaKind { MEDIA_TEXT = 0, MEDIA_IMAGE = 1, MEDIA_AUDIO = 2 };
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
    // GPT: text (characters or subword pieces), images (one token per pixel
    // colour, row by row) or sound (one token per mu-law sample)
    int vocab = 0, ctx = 64, dim = 64, heads = 4, nlayer = 2;
    int media = MEDIA_TEXT;
    int img_size = 0;  // images: width = height in pixels
    int rate = 0;      // sound: samples per second
    // LoRA rank of adapters attached to every Linear (0 = none)
    int lora_rank = 0;

    std::string to_text() const;
    bool from_text(const std::string& s);
};

size_t estimate_params(const ModelSpec& s);
// Rough peak memory (bytes) needed to train with the given batch size.
size_t estimate_train_bytes(const ModelSpec& s, int batch);

// One mini-batch. MLP uses x (+ tgt or y); GPT uses ids + tgt with B, T.
// w optionally holds a loss weight per target (empty = all weigh the same).
struct Batch {
    Tensor x, y;
    std::vector<int> ids, tgt;
    std::vector<float> w;
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
    // Notes kept in the file, e.g. the dialogue tags "q_tag" and "a_tag" of
    // the text the model learned (used by the chat page).
    std::map<std::string, std::string> meta;

    explicit Model(const ModelSpec& s, uint64_t seed = 1234);

    // Forward pass, loss, and (if train) backward pass that ACCUMULATES
    // gradients scaled by `scale`. Returns the mean loss.
    float step(const Batch& b, bool train, float scale, int* correct = nullptr);
    // Text models run step() entirely on the GPU while gpu_use() is on, unless this is false.
    bool use_gpu = true;
    bool last_step_on_gpu = false;
    void zero_grad();
    std::vector<Param*> params();
    size_t param_count(bool trainable_only = false);

    void enable_lora(int rank);
    void merge_lora();
    // Dropout rate of the transformer blocks while training (text models).
    void set_dropout(float p);
    // Text models: appends the tokens the model does not know yet (for
    // fine-tuning on data with new characters). Not possible while LoRA is on.
    // Returns how many were added.
    int extend_vocab(const std::vector<std::string>& tokens);

    // MLP inference on raw (unnormalized) features: class probabilities, or
    // a single de-normalized value for regression.
    std::vector<float> predict(const std::vector<float>& features);
    // GPT inference: samples the next token id given previous ids.
    int next_token(const std::vector<int>& ids, float temperature, Rng& rng);
    // GPT inference with cached keys and values (much faster for long outputs):
    // continues `prompt` by up to n tokens, stopping after `stop` (if >= 0).
    // `on_token` (optional) sees each new token and can return false to stop.
    std::vector<int> generate(const std::vector<int>& prompt, int n, float temperature, Rng& rng, int stop = -1,
                              const std::function<bool(int)>& on_token = nullptr);
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
    // incremental generation
    const float* gen_feed(int token);
    int sample(const float* z, float temperature, Rng& r) const;
    std::vector<Tensor> kv_;
    int gen_pos_ = 0;
    Tensor gx_, gy_, gf_, glog_;
    // scratch
    Tensor dout, d1, d2;
    Batch infer;
};

// A question test for text models: the model continues `prompt` (greedy,
// up to the end of the line) and the result is compared with `expected`.
// `question` is the human-readable form shown in logs and reports.
struct Probe {
    std::string prompt, expected, question;
};
struct ProbeResult {
    std::string answer;
    bool pass = false;
};
ProbeResult run_probe(Model& m, const Probe& p);

struct Image;
struct Sound;
// The pixels of an image model's tokens ("#rrggbb" palette entries).
Image image_from_tokens(const std::vector<std::string>& vocab, const int* toks, size_t n, int size);
// A new image from an image model (MEDIA_IMAGE).
Image sample_image(Model& m, float temperature, Rng& rng);
// New sound from a sound model (MEDIA_AUDIO). `progress` (optional) is called
// with the share done so far and can return false to stop early.
Sound sample_sound(Model& m, float seconds, float temperature, Rng& rng,
                   const std::function<bool(float)>& progress = nullptr);

uint16_t float_to_half(float f);
float half_to_float(uint16_t h);
// Splits UTF-8 text into code points (as strings); drops '\r'.
std::vector<std::string> utf8_chars(const std::string& s);
// Text to token ids: at each position the longest vocabulary entry that
// matches (single characters for character-level models). Characters the
// vocabulary does not cover are skipped.
std::vector<int> tokenize(const std::string& text, const std::vector<std::string>& vocab);

}  // namespace kor
