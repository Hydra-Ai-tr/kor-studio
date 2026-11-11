// Kor Studio - GGUF language models (the llama.cpp file format): Llama, Mistral
// and Qwen2 style models read in place from the file, with their tokenizer and
// chat format, and LoRA fine-tuning on top of the frozen (quantized) weights.
#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "model.h"
#include "tensor.h"

namespace kor {

enum GgmlType {
    GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q4_1 = 3, GGML_Q5_0 = 6, GGML_Q5_1 = 7, GGML_Q8_0 = 8,
    GGML_Q2_K = 10, GGML_Q3_K = 11, GGML_Q4_K = 12, GGML_Q5_K = 13, GGML_Q6_K = 14, GGML_BF16 = 30,
};
const char* ggml_type_name(int type);  // nullptr if the type is not supported
int ggml_block_size(int type);         // values per block
size_t ggml_block_bytes(int type);
size_t ggml_row_bytes(int type, int64_t n);
// Packs n values (a multiple of the block size) as F32, F16, Q8_0 or Q4_0.
bool ggml_quantize(int type, const float* x, int64_t n, std::vector<uint8_t>& out);
void ggml_dequantize(int type, const uint8_t* src, int64_t n, float* out);

// A weight matrix as stored in the file: `rows` rows of `cols` values (a GGUF
// tensor with dims {cols, rows}), used without unpacking it as a whole.
struct QMat {
    int type = GGML_F32;
    int rows = 0, cols = 0;
    const uint8_t* data = nullptr;
    size_t row_bytes = 0;
    void row(int r, float* out) const;
    // Y [n x rows] = X [n x cols] * W^T (+ b)
    void mul(const float* X, int n, float* Y, const float* b = nullptr) const;
    // dX [n x cols] += dY [n x rows] * W
    void mul_back(const float* dY, int n, float* dX) const;
};

// Writes GGUF files (model exports for tests, LoRA adapters).
class GgufWriter {
public:
    void set_str(const std::string& key, const std::string& v);
    void set_u32(const std::string& key, uint32_t v);
    void set_f32(const std::string& key, float v);
    void set_bool(const std::string& key, bool v);
    void set_strs(const std::string& key, const std::vector<std::string>& v);
    void set_i32s(const std::string& key, const std::vector<int32_t>& v);
    void set_f32s(const std::string& key, const std::vector<float>& v);
    // dims as in GGUF: dims[0] is the row length.
    void add_tensor(const std::string& name, std::vector<int64_t> dims, int type, std::vector<uint8_t> data);
    bool save(const std::string& path, std::string* err = nullptr) const;

private:
    struct Kv { std::string key; std::vector<uint8_t> bytes; };
    struct Tn { std::string name; std::vector<int64_t> dims; int type; std::vector<uint8_t> data; };
    std::vector<Kv> kvs_;
    std::vector<Tn> tns_;
};

struct LlmMsg { std::string role, text; };  // role: "system", "user" or "assistant"

// LoRA training settings for GgufModel::lora_train.
struct LoraConfig {
    int rank = 8;
    float alpha = 16.f;           // the update is scaled by alpha / rank
    bool mlp = false;             // also adapt the feed-forward matrices (gate, up, down)
    int steps = 200, batch = 4, seq = 256;
    float lr = 2e-4f, weight_decay = 0.f;
    int eval_every = 25;
    uint64_t seed = 1;
};

class GgufModel {
public:
    struct Config {
        std::string arch;
        int vocab = 0, ctx_train = 0, dim = 0, layers = 0, ffn = 0, heads = 0, kv_heads = 0, head_dim = 0;
        float rope_base = 10000.f, eps = 1e-5f;
        bool rope_neox = false;  // rotates (i, i + d/2) pairs instead of (2i, 2i + 1)
    } cfg;
    std::string path, name, quant;  // quant: the type most weights are stored in
    std::string quant_mix;          // e.g. "Q4_K 80%, Q6_K 20%"
    size_t file_bytes = 0, n_params = 0;
    int ctx = 0;  // the cache length used here

    static std::shared_ptr<GgufModel> load(const std::string& path, std::string* err = nullptr, int max_ctx = 4096);
    ~GgufModel();

    // ---- tokens
    std::vector<int> encode(const std::string& text, bool add_bos, bool special = true) const;
    std::string piece(int id) const;  // the bytes a token stands for ("" for control tokens)
    std::string decode(const std::vector<int>& ids) const;
    bool is_end(int id) const;        // end of text or of a turn
    int eos() const;
    std::string chat_format() const;  // "ChatML", "Llama 3", ...
    // A conversation in the model's chat format; with open_answer it ends where
    // the assistant's next answer starts.
    std::string chat_text(const std::vector<LlmMsg>& msgs, bool open_answer = true) const;
    std::vector<int> chat_ids(const std::vector<LlmMsg>& msgs, bool open_answer = true) const;

    // ---- inference
    // Runs n tokens at positions pos0.. and keeps their keys and values in the
    // cache; returns the logits of the last one.
    const float* eval(const int* toks, int n, int pos0);
    // Continues `prompt` by up to n tokens (stopping at an end token), reusing the
    // cache for the part it shares with the previous call. on_token gets every new
    // token and returns false to stop. Returns the number of tokens made.
    int generate(const std::vector<int>& prompt, int n, const Sampling& s, Rng& rng,
                 const std::function<bool(int)>& on_token);
    // generate() with the answer streamed as whole UTF-8 characters; *tokens gets
    // the number of tokens made.
    std::string reply(const std::vector<int>& prompt, int n, const Sampling& s, Rng& rng,
                      const std::function<bool(const std::string&)>& on_text, int* tokens = nullptr);
    // The conversation as prompt tokens, dropping the oldest turns (never the
    // system message or the last one) so that `room` tokens are left for the answer.
    std::vector<int> chat_prompt(const std::vector<LlmMsg>& msgs, int room) const;

    // ---- LoRA: small trainable updates to the attention (and optionally
    // feed-forward) matrices; the file's weights never change.
    bool has_lora() const;
    void lora_init(const LoraConfig& c);
    void lora_clear();
    // Mean cross-entropy of the targets (weight 0 = not counted) of B sequences of T tokens.
    float lora_loss(const int* ids, const int* targets, const float* w, int B, int T, bool backward);
    std::vector<Param*> lora_params();
    // Saves / loads the adapter as a llama.cpp compatible GGUF LoRA file.
    bool lora_save(const std::string& path, std::string* err = nullptr) const;
    bool lora_load(const std::string& path, std::string* err = nullptr);
    LoraConfig lora_cfg;

private:
    GgufModel();
    struct Impl;
    std::unique_ptr<Impl> d;
};

// Turns a training text into fine-tuning examples: with question/answer tags each
// pair becomes one chat turn (only the answer is learned), else plain text.
struct LoraData {
    std::vector<std::vector<int>> ex;    // token sequences
    std::vector<std::vector<float>> w;   // per token: how much predicting it counts
    bool chat = false;                   // one example per question/answer pair
    size_t tokens() const {
        size_t n = 0;
        for (const auto& e : ex) n += e.size();
        return n;
    }
};
LoraData lora_data(const GgufModel& m, const std::string& text, const std::string& q_tag, const std::string& a_tag,
                   const std::string& system = "");

struct LoraProgress {
    int step = 0, steps = 0;
    float loss = 0, val = -1, tok_s = 0;  // val: the latest validation loss
    bool evaluated = false;                // val was computed at this step
};
// Trains the model's LoRA adapter on `data` (10% held out when it is long enough).
// on_step returns false to stop. Returns false with *err on failure.
bool lora_train(GgufModel& m, const LoraData& data, const LoraConfig& c,
                const std::function<bool(const LoraProgress&)>& on_step, std::string* err = nullptr);

}  // namespace kor
