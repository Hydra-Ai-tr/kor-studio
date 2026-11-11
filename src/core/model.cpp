#include "model.h"

#include "gpu.h"
#include "media.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace kor {

std::filesystem::path upath(const std::string& utf8) { return std::filesystem::u8path(utf8); }

// ------------------------------------------------------------- ModelSpec

std::string ModelSpec::to_text() const {
    std::ostringstream o;
    o << "kind=" << kind << "\nin_dim=" << in_dim << "\nout_dim=" << out_dim
      << "\nregression=" << (regression ? 1 : 0) << "\nvocab=" << vocab << "\nctx=" << ctx
      << "\ndim=" << dim << "\nheads=" << heads << "\nnlayer=" << nlayer
      << "\nlora_rank=" << lora_rank << "\nmedia=" << media << "\nimg_size=" << img_size
      << "\nrate=" << rate;
    if (arch != ARCH_CLASSIC) o << "\narch=" << arch << "\nffn=" << ffn << "\nrope_theta=" << rope_theta;
    o << "\nlayers=";
    for (size_t i = 0; i < layers.size(); ++i)
        o << (i ? "," : "") << layers[i].type << ":" << layers[i].units << ":" << layers[i].p;
    o << "\n";
    return o.str();
}

bool ModelSpec::from_text(const std::string& s) {
    std::istringstream in(s);
    std::string line;
    layers.clear();
    while (std::getline(in, line)) {
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        if (k == "layers") {
            std::istringstream ls(v);
            std::string item;
            while (std::getline(ls, item, ',')) {
                LayerSpec l;
                char c1, c2;
                std::istringstream is(item);
                if (is >> l.type >> c1 >> l.units >> c2 >> l.p) layers.push_back(l);
            }
            continue;
        }
        int iv = std::atoi(v.c_str());
        if (k == "kind") kind = iv;
        else if (k == "in_dim") in_dim = iv;
        else if (k == "out_dim") out_dim = iv;
        else if (k == "regression") regression = iv != 0;
        else if (k == "vocab") vocab = iv;
        else if (k == "ctx") ctx = iv;
        else if (k == "dim") dim = iv;
        else if (k == "heads") heads = iv;
        else if (k == "nlayer") nlayer = iv;
        else if (k == "lora_rank") lora_rank = iv;
        else if (k == "media") media = iv;
        else if (k == "img_size") img_size = iv;
        else if (k == "rate") rate = iv;
        else if (k == "arch") arch = iv;
        else if (k == "ffn") ffn = iv;
        else if (k == "rope_theta") rope_theta = (float)std::atof(v.c_str());
    }
    if (kind == MODEL_GPT) {
        if (arch != ARCH_CLASSIC && arch != ARCH_MODERN) return false;
        if (arch == ARCH_MODERN && ((dim / std::max(1, heads)) % 2 != 0 || ffn < 0 || !(rope_theta > 1.f))) return false;
        return vocab > 0 && ctx > 0 && dim > 0 && heads > 0 && dim % heads == 0 && nlayer > 0;
    }
    return in_dim > 0 && out_dim > 0;
}

int default_ffn(int dim) { return std::max(32, (8 * dim / 3 + 31) / 32 * 32); }

int ModelSpec::mlp_hidden() const {
    if (arch != ARCH_MODERN) return 4 * dim;
    return ffn > 0 ? ffn : default_ffn(dim);
}

size_t estimate_params(const ModelSpec& s) {
    size_t n = 0;
    if (s.kind == MODEL_GPT) {
        size_t C = s.dim, V = s.vocab, F = s.mlp_hidden(), F1 = s.mlp_in_width();
        bool modern = s.arch == ARCH_MODERN;
        n += V * C + (modern ? 0 : (size_t)s.ctx * C);
        size_t norms = modern ? 2 * C : 4 * C;
        size_t blk = norms + (C * 3 * C + 3 * C) + (C * C + C) + (C * F1 + F1) + (F * C + C);
        n += blk * s.nlayer + (modern ? C : 2 * C) + C * V + V;
    } else {
        size_t cur = s.in_dim;
        for (auto& l : s.layers) {
            if (l.type == L_DENSE) { n += cur * l.units + l.units; cur = l.units; }
            else if (l.type == L_LAYERNORM) n += 2 * cur;
        }
        n += cur * s.out_dim + s.out_dim;
    }
    return n;
}

size_t estimate_train_bytes(const ModelSpec& s, int batch) {
    size_t bytes = estimate_params(s) * 4 * 4;  // value + gradient + two Adam moments
    if (s.kind == MODEL_GPT) {
        size_t N = (size_t)batch * s.ctx, C = s.dim, F = s.mlp_hidden(), F1 = s.mlp_in_width();
        size_t per_block = N * (5 * C + 2 * F1 + F) * 2 + (size_t)batch * s.heads * s.ctx * s.ctx;
        bytes += (per_block * s.nlayer + N * C * 4 + N * (size_t)s.vocab * 2) * 4;
    } else {
        size_t width = s.in_dim + s.out_dim;
        size_t cur = s.in_dim;
        for (auto& l : s.layers) {
            if (l.type == L_DENSE) cur = l.units;
            width += cur;
        }
        bytes += (size_t)batch * width * 3 * 4;
    }
    return bytes;
}

// ------------------------------------------------------------ half float

uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int e = (int)((x >> 23) & 0xff);
    uint32_t man = x & 0x7fffffu;
    if (e == 255) return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0u));
    int exp = e - 127 + 15;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t h = man >> shift;
        if ((man >> (shift - 1)) & 1u) ++h;
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)exp << 10) | (man >> 13);
    if (man & 0x1000u) ++h;  // may carry into the exponent, which is correct
    return (uint16_t)(sign | h);
}

float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t x;
    if (exp == 0) {
        if (man == 0) x = sign;
        else {
            int e = -1;
            do { ++e; man <<= 1; } while (!(man & 0x400u));
            man &= 0x3ffu;
            x = sign | ((uint32_t)(127 - 15 - e) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7f800000u | (man << 13);
    } else {
        x = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

size_t utf8_len(const char* s, size_t i, size_t n) {
    unsigned char c = (unsigned char)s[i];
    size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
    return i + len > n ? 1 : len;
}

std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0, n = s.size();
    while (i < n) {
        size_t len = utf8_len(s.data(), i, n);
        if (!(len == 1 && s[i] == '\r')) out.emplace_back(s, i, len);
        i += len;
    }
    return out;
}

// ----------------------------------------------------------------- Model

Model::Model(const ModelSpec& s, uint64_t seed) : spec(s), rng(seed) {
    if (spec.kind == MODEL_GPT) {
        bool modern = spec.arch == ARCH_MODERN;
        emb.reset(new Embedding(spec.vocab, spec.ctx, spec.dim, rng, !modern));
        float resid_std = 0.02f / std::sqrt(2.f * (float)spec.nlayer);
        int ffn = modern ? spec.mlp_hidden() : 0;
        for (int i = 0; i < spec.nlayer; ++i) {
            blocks.emplace_back(new Block(spec.dim, spec.heads, "h" + std::to_string(i), rng, resid_std, ffn, spec.rope_theta));
            blocks.back()->linears(all_linears);
        }
        lnf.reset(new LayerNorm(spec.dim, "lnf", modern));
        head.reset(new Linear(spec.dim, spec.vocab, "head", rng, 0.02f));
        all_linears.push_back(head.get());
        xs.resize(spec.nlayer + 1);
    } else {
        int cur = spec.in_dim, idx = 0;
        for (auto& l : spec.layers) {
            std::string name = "L" + std::to_string(idx++);
            switch (l.type) {
                case L_DENSE: {
                    Linear* lin = new Linear(cur, l.units, name, rng);
                    layers.emplace_back(lin);
                    all_linears.push_back(lin);
                    cur = l.units;
                    break;
                }
                case L_RELU: layers.emplace_back(new Activation(ACT_RELU)); break;
                case L_GELU: layers.emplace_back(new Activation(ACT_GELU)); break;
                case L_TANH: layers.emplace_back(new Activation(ACT_TANH)); break;
                case L_SIGMOID: layers.emplace_back(new Activation(ACT_SIGMOID)); break;
                case L_DROPOUT: layers.emplace_back(new Dropout(l.p, rng.next())); break;
                case L_LAYERNORM: layers.emplace_back(new LayerNorm(cur, name)); break;
                default: break;
            }
        }
        Linear* out = new Linear(cur, spec.out_dim, "out", rng, std::sqrt(1.f / (float)cur));
        layers.emplace_back(out);
        all_linears.push_back(out);
        if (Linear* first = dynamic_cast<Linear*>(layers[0].get())) first->need_dx = false;
        acts.resize(layers.size());
    }
    if (spec.lora_rank > 0) {
        int r = spec.lora_rank;
        spec.lora_rank = 0;
        enable_lora(r);
    }
}

std::vector<Param*> Model::params() {
    std::vector<Param*> ps;
    if (spec.kind == MODEL_GPT) {
        emb->collect(ps);
        for (auto& b : blocks) b->collect(ps);
        lnf->collect(ps);
        head->collect(ps);
    } else {
        for (auto& l : layers) l->collect(ps);
    }
    return ps;
}

size_t Model::param_count(bool trainable_only) {
    size_t n = 0;
    for (Param* p : params())
        if (!trainable_only || p->trainable) n += p->w.size();
    return n;
}

void Model::zero_grad() {
    for (Param* p : params())
        if (p->trainable) p->g.zero();
}

void Model::enable_lora(int rank) {
    if (rank <= 0 || spec.lora_rank > 0) return;
    for (Param* p : params()) p->freeze();  // only the adapters will train
    for (Linear* l : all_linears) l->enable_lora(rank, rng);
    spec.lora_rank = rank;
}

int Model::extend_vocab(const std::vector<std::string>& tokens) {
    if (spec.kind != MODEL_GPT || spec.lora_rank > 0) return 0;
    std::vector<std::string> add;
    for (auto& t : tokens)
        if (std::find(vocab.begin(), vocab.end(), t) == vocab.end() && std::find(add.begin(), add.end(), t) == add.end())
            add.push_back(t);
    if (add.empty()) return 0;
    int C = spec.dim, V0 = spec.vocab, V1 = V0 + (int)add.size();
    Param& tok = emb->tok;
    tok.w.v.resize((size_t)V1 * C);
    tok.w.rows = V1;
    for (size_t i = (size_t)V0 * C; i < tok.w.v.size(); ++i) tok.w.v[i] = rng.normal() * 0.02f;
    Tensor W(C, V1);
    for (int r = 0; r < C; ++r) {
        std::copy(head->W.w.row(r), head->W.w.row(r) + V0, W.row(r));
        for (int j = V0; j < V1; ++j) W.row(r)[j] = rng.normal() * 0.02f;
    }
    head->W.w = std::move(W);
    // New tokens start less likely than the least likely known one.
    float low = *std::min_element(head->b.w.v.begin(), head->b.w.v.end());
    head->b.w.v.resize(V1, low - 2.f);
    head->b.w.cols = V1;
    for (Param* p : {&tok, &head->W, &head->b}) {
        p->g.resize(p->w.rows, p->w.cols);
        p->m = Tensor();
        p->s = Tensor();
    }
    head->out = V1;
    emb->vocab = V1;
    spec.vocab = V1;
    vocab.insert(vocab.end(), add.begin(), add.end());
    return (int)add.size();
}

void Model::release_scratch() {
    for (auto& b : blocks) b->release_scratch();
    if (lnf) lnf->release_scratch();
    if (head) head->release_scratch();
    for (Tensor& t : xs) t = Tensor();
    for (Tensor& t : acts) t = Tensor();
    xf = logits = dout = d1 = d2 = Tensor();
}

bool Model::gpu_layout(GpuGpt& g) {
    if (spec.kind != MODEL_GPT || spec.lora_rank != 0) return false;
    g.vocab = head->out; g.ctx = spec.ctx; g.dim = spec.dim; g.heads = spec.heads; g.nlayer = spec.nlayer;
    g.modern = spec.arch == ARCH_MODERN;
    g.ffn = spec.mlp_hidden();
    g.rope_theta = spec.rope_theta;
    g.recompute = recompute ? 1 : -1;
    g.params = params();
    return true;
}

void Model::set_dropout(float p) {
    p = std::max(0.f, std::min(0.9f, p));
    for (auto& b : blocks) b->set_dropout(p);
}

void Model::merge_lora() {
    if (spec.lora_rank <= 0) return;
    for (Linear* l : all_linears) l->merge_lora();
    for (Param* p : params()) {
        p->trainable = true;
        p->g.resize(p->w.rows, p->w.cols);
    }
    spec.lora_rank = 0;
}

void Model::forward(const Batch& b, bool train) {
    if (spec.kind == MODEL_GPT) {
        emb->forward(b.ids, b.B, b.T, xs[0]);
        recomputing_ = train && recompute;
        if (recomputing_) drop_rng_.resize(spec.nlayer);
        for (int l = 0; l < spec.nlayer; ++l) {
            if (recomputing_) drop_rng_[l] = {blocks[l]->drop1.rng, blocks[l]->drop2.rng};
            blocks[l]->forward(xs[l], xs[l + 1], b.B, b.T, train);
            if (recomputing_) blocks[l]->release_scratch();
        }
        lnf->forward(xs[spec.nlayer], xf, train);
        head->forward(xf, logits, train);
    } else {
        const Tensor* cur = &b.x;
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i]->forward(*cur, acts[i], train);
            cur = &acts[i];
        }
    }
}

const Tensor& Model::output() const { return spec.kind == MODEL_GPT ? logits : acts.back(); }

float Model::step(const Batch& b, bool train, float scale, int* correct) {
    last_step_on_gpu = false;
    GpuGpt g;
    if (use_gpu && gpu_active() && (int)b.ids.size() == b.B * b.T && gpu_layout(g)) {
        const float* w = b.w.size() == b.tgt.size() && !b.w.empty() ? b.w.data() : nullptr;
        float p = blocks.empty() ? 0.f : blocks[0]->drop1.p, loss = 0;
        if (gpu_gpt_step(g, b.ids.data(), b.tgt.data(), w, b.B, b.T, train, p, rng.next(), scale, &loss, correct)) {
            last_step_on_gpu = true;
            return loss;
        }
    }
    forward(b, train);
    const Tensor& out = output();
    float loss;
    if (spec.kind == MODEL_MLP && spec.regression) {
        loss = mse_loss(out, b.y, train ? &dout : nullptr, scale);
        if (correct) *correct = 0;
    } else {
        const float* w = b.w.size() == b.tgt.size() && !b.w.empty() ? b.w.data() : nullptr;
        loss = cross_entropy(out, b.tgt.data(), train ? &dout : nullptr, scale, correct, w);
    }
    if (!train) return loss;
    if (spec.kind == MODEL_GPT) {
        head->backward(dout, d1);
        lnf->backward(d1, d2);
        for (int l = spec.nlayer - 1; l >= 0; --l) {
            if (recomputing_) {
                // the same masks as the first time; xs[l + 1] gets the values it already holds
                blocks[l]->drop1.rng = drop_rng_[l].first;
                blocks[l]->drop2.rng = drop_rng_[l].second;
                blocks[l]->forward(xs[l], xs[l + 1], b.B, b.T, true);
            }
            blocks[l]->backward(d2, d1);
            if (recomputing_) blocks[l]->release_scratch();
            std::swap(d1, d2);
        }
        emb->backward(d2);
    } else {
        Tensor* a = &dout;
        Tensor* bufs[2] = {&d1, &d2};
        int k = 0;
        for (int i = (int)layers.size() - 1; i >= 0; --i) {
            layers[i]->backward(*a, *bufs[k]);
            a = bufs[k];
            k ^= 1;
        }
    }
    return loss;
}

std::vector<float> Model::predict(const std::vector<float>& features) {
    std::vector<float> res;
    if (spec.kind != MODEL_MLP || (int)features.size() != spec.in_dim) return res;
    infer.x.resize(1, spec.in_dim);
    for (int j = 0; j < spec.in_dim; ++j) {
        float m = j < (int)norm_mean.size() ? norm_mean[j] : 0.f;
        float s = j < (int)norm_std.size() ? norm_std[j] : 1.f;
        infer.x.v[j] = (features[j] - m) / s;
    }
    forward(infer, false);
    const Tensor& out = output();
    if (spec.regression) {
        float m = (int)norm_mean.size() > spec.in_dim ? norm_mean[spec.in_dim] : 0.f;
        float s = (int)norm_std.size() > spec.in_dim ? norm_std[spec.in_dim] : 1.f;
        res.push_back(out.v[0] * s + m);
    } else {
        float mx = out.v[0];
        for (float z : out.v) mx = std::max(mx, z);
        float sum = 0;
        for (float z : out.v) { res.push_back(std::exp(z - mx)); sum += res.back(); }
        for (float& p : res) p /= sum;
    }
    return res;
}

int Model::next_token(const std::vector<int>& ids, float temperature, Rng& r) {
    if (spec.kind != MODEL_GPT) return 0;
    int n = std::min((int)ids.size(), spec.ctx);
    if (n == 0) { infer.ids.assign(1, 0); n = 1; }
    else infer.ids.assign(ids.end() - n, ids.end());
    infer.B = 1; infer.T = n;
    forward(infer, false);
    return sample(logits.row(n - 1), temperature, r);
}

int Model::sample(const float* z0, const Sampling& s, const int* recent, size_t n_recent, Rng& r, int spare) const {
    return sample_logits(z0, spec.vocab, s, recent, n_recent, r, spare);
}

static int sample_plain(const float* z, int V, float temperature, Rng& r) {
    int arg = 0;
    for (int j = 1; j < V; ++j)
        if (z[j] > z[arg]) arg = j;
    if (temperature <= 0.01f) return arg;
    std::vector<double> p(V);
    double sum = 0;
    for (int j = 0; j < V; ++j) { p[j] = std::exp((double)(z[j] - z[arg]) / temperature); sum += p[j]; }
    double u = r.uniform() * sum;
    for (int j = 0; j < V; ++j) { u -= p[j]; if (u <= 0) return j; }
    return arg;
}

int sample_logits(const float* z0, int V, const Sampling& s, const int* recent, size_t n_recent, Rng& r, int spare) {
    bool plain = s.top_k <= 0 && s.top_p >= 1.f && (s.repeat_penalty == 1.f || n_recent == 0);
    if (plain) return sample_plain(z0, V, s.temperature, r);
    std::vector<float> z(z0, z0 + V);
    if (s.repeat_penalty != 1.f && s.repeat_penalty > 0.f) {
        std::vector<char> seen(V, 0);
        size_t from = n_recent > (size_t)std::max(0, s.repeat_window) ? n_recent - (size_t)std::max(0, s.repeat_window) : 0;
        for (size_t i = from; i < n_recent; ++i)
            if (recent[i] >= 0 && recent[i] < V && recent[i] != spare && !seen[recent[i]]) {
                seen[recent[i]] = 1;
                float& v = z[recent[i]];
                v = v > 0 ? v / s.repeat_penalty : v * s.repeat_penalty;
            }
    }
    int arg = 0;
    for (int j = 1; j < V; ++j)
        if (z[j] > z[arg]) arg = j;
    if (s.temperature <= 0.01f) return arg;
    std::vector<int> order(V);
    for (int j = 0; j < V; ++j) order[j] = j;
    int keep = s.top_k > 0 ? std::min(V, s.top_k) : V;
    auto by_logit = [&](int a, int b) { return z[a] > z[b] || (z[a] == z[b] && a < b); };
    if (keep < V) std::partial_sort(order.begin(), order.begin() + keep, order.end(), by_logit);
    else if (s.top_p < 1.f) std::sort(order.begin(), order.end(), by_logit);
    std::vector<double> p(keep);
    double sum = 0;
    for (int i = 0; i < keep; ++i) { p[i] = std::exp((double)(z[order[i]] - z[arg]) / s.temperature); sum += p[i]; }
    if (s.top_p < 1.f) {
        double need = std::max(0.f, s.top_p) * sum, acc = 0;
        int n = 0;
        while (n < keep && (n == 0 || acc < need)) acc += p[n++];
        keep = n;
        sum = acc;
    }
    double u = r.uniform() * sum;
    for (int i = 0; i < keep; ++i) { u -= p[i]; if (u <= 0) return order[i]; }
    return order[keep - 1];
}

int Model::sample(const float* z, float temperature, Rng& r) const { return sample_plain(z, spec.vocab, temperature, r); }

const float* Model::gen_feed(int token) {
    if (gen_gpu_) {
        const float* z = gpu_gen_feed(token, gen_pos_);
        if (z) ++gen_pos_;
        return z;  // nullptr: the caller goes on on the CPU
    }
    int C = spec.dim;
    if ((int)kv_.size() != spec.nlayer) kv_.resize(spec.nlayer);
    for (auto& t : kv_)
        if (t.rows != spec.ctx || t.cols != 2 * C) t.resize(spec.ctx, 2 * C);
    gx_.resize_noinit(1, C);
    const float* a = emb->tok.w.row(token);
    if (emb->learned_pos) {
        const float* p = emb->pos.w.row(gen_pos_);
        for (int j = 0; j < C; ++j) gx_.v[j] = a[j] + p[j];
    } else {
        std::memcpy(gx_.v.data(), a, sizeof(float) * C);
    }
    for (int l = 0; l < spec.nlayer; ++l) {
        blocks[l]->step(gx_, gy_, kv_[l], gen_pos_);
        std::swap(gx_, gy_);
    }
    lnf->forward(gx_, gf_, false);
    head->forward(gf_, glog_, false);
    ++gen_pos_;
    return glog_.row(0);
}

std::vector<int> Model::generate(const std::vector<int>& prompt, int n, float temperature, Rng& rng, int stop,
                                 const std::function<bool(int)>& on_token) {
    Sampling s;
    s.temperature = temperature;
    return generate(prompt, n, s, rng, stop, on_token);
}

std::vector<int> Model::generate(const std::vector<int>& prompt, int n, const Sampling& s, Rng& rng, int stop,
                                 const std::function<bool(int)>& on_token) {
    std::vector<int> out;
    if (spec.kind != MODEL_GPT || n <= 0) return out;
    std::vector<int> hist = prompt;
    if (hist.empty()) hist.push_back(0);
    int ctx = spec.ctx;
    GpuGpt gl;
    gen_gpu_ = gen_on_gpu && gpu_layout(gl) && gpu_gen_begin(gl);
    // Starts a fresh window holding the last `keep` tokens of the history; a
    // GPU error starts it again on the CPU.
    std::function<const float*(size_t)> prime = [&](size_t keep) -> const float* {
        gen_pos_ = 0;
        keep = std::min(keep, hist.size());
        const float* z = nullptr;
        for (size_t i = hist.size() - keep; i < hist.size(); ++i) {
            z = gen_feed(hist[i]);
            if (!z) {
                gen_gpu_ = false;
                return prime(keep);
            }
        }
        return z;
    };
    auto feed = [&](int t) {
        size_t window = (size_t)gen_pos_ + 1;
        const float* z = gen_feed(t);
        if (z) return z;
        gen_gpu_ = false;
        return prime(window);
    };
    const float* z = prime((size_t)ctx);
    for (int i = 0; i < n; ++i) {
        size_t look = std::min(hist.size(), (size_t)std::max(0, s.repeat_window));
        int t = sample(z, s, hist.data() + hist.size() - look, look, rng, stop);
        out.push_back(t);
        hist.push_back(t);
        if (t == stop || (on_token && !on_token(t))) break;
        if (gen_pos_ >= ctx) {
            z = prime((size_t)std::max(1, ctx / 2));
            if (hist.size() > (size_t)ctx * 4) hist.erase(hist.begin(), hist.end() - ctx);
        } else {
            z = feed(t);
        }
    }
    last_gen_on_gpu = gen_gpu_;
    gen_gpu_ = false;
    return out;
}

std::vector<int> Model::encode(const std::string& text) const { return tokenize(text, vocab); }

Tokenizer::Tokenizer(const std::vector<std::string>& vocab) : root_(256, -1) {
    nodes_.push_back(-1);
    for (size_t v = 0; v < vocab.size(); ++v) {
        const std::string& p = vocab[v];
        if (p.find('\r') != std::string::npos) continue;  // the text is read without '\r' (see utf8_chars)
        int node = 0;
        for (size_t i = 0; i < p.size(); ++i) {
            unsigned char b = (unsigned char)p[i];
            int& next = node == 0 ? root_[b] : kids_.emplace((uint64_t)node << 8 | b, -1).first->second;
            if (next < 0) {
                next = (int)nodes_.size();
                nodes_.push_back(-1);
            }
            node = next;
        }
        if (node > 0 && nodes_[node] < 0) nodes_[node] = (int)v;  // the first of equal pieces wins
    }
}

void Tokenizer::encode(const char* s, size_t n, std::vector<int>& out) const {
    size_t i = 0;
    while (i < n) {
        // longest piece from i, extended one whole character at a time
        int best = -1, node = 0;
        size_t best_end = 0, j = i, first_end = 0;
        while (j < n) {
            size_t len = utf8_len(s, j, n);
            if (len == 1 && s[j] == '\r') {
                j += 1;
                if (first_end == 0 && node == 0) i = j;  // a leading '\r' is not a character
                continue;
            }
            for (size_t k = 0; k < len && node >= 0; ++k) {
                unsigned char b = (unsigned char)s[j + k];
                if (node == 0) node = root_[b];
                else {
                    auto it = kids_.find((uint64_t)node << 8 | b);
                    node = it == kids_.end() ? -1 : it->second;
                }
            }
            j += len;
            if (first_end == 0) first_end = j;
            if (node < 0) break;
            if (nodes_[node] >= 0) { best = nodes_[node]; best_end = j; }
        }
        if (first_end == 0) break;  // only '\r' was left
        if (best >= 0) out.push_back(best);  // unknown characters are skipped
        i = best >= 0 ? best_end : first_end;
    }
}

std::vector<int> tokenize(const std::string& text, const std::vector<std::string>& vocab) {
    std::vector<int> ids;
    ids.reserve(text.size() / 2);
    Tokenizer(vocab).encode(text.data(), text.size(), ids);
    ids.shrink_to_fit();
    return ids;
}

static std::string trim_ws(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

ProbeResult run_probe(Model& m, const Probe& p) {
    ProbeResult res;
    if (m.spec.kind != MODEL_GPT) return res;
    std::vector<int> ids = m.encode(p.prompt);
    if (ids.empty()) return res;
    int newline = -1;
    for (size_t i = 0; i < m.vocab.size(); ++i)
        if (m.vocab[i] == "\n") newline = (int)i;
    int max_len = std::min(160, (int)utf8_chars(p.expected).size() * 2 + 16);
    Rng rng(1);
    // temperature 0: always the most likely token
    for (int t : m.generate(ids, max_len, 0.f, rng, newline)) {
        if (t == newline) break;
        res.answer += m.vocab[t];
    }
    res.answer = trim_ws(res.answer);
    res.pass = !res.answer.empty() && res.answer == trim_ws(p.expected);
    return res;
}

// ------------------------------------------------------- images and sound

Image image_from_tokens(const std::vector<std::string>& vocab, const int* toks, size_t n, int size) {
    std::vector<std::array<uint8_t, 3>> pal(vocab.size(), {{0, 0, 0}});
    for (size_t i = 0; i < vocab.size(); ++i) {
        unsigned r = 0, g = 0, b = 0;
        if (std::sscanf(vocab[i].c_str(), "#%02x%02x%02x", &r, &g, &b) == 3) pal[i] = {{(uint8_t)r, (uint8_t)g, (uint8_t)b}};
    }
    Image img;
    img.w = img.h = size;
    img.rgb.assign((size_t)size * size * 3, 0);
    std::array<uint8_t, 3> last = {{0, 0, 0}};
    for (size_t i = 0; i < (size_t)size * size; ++i) {
        // A separator in the middle of an image keeps the previous colour.
        if (i < n && toks[i] > 0 && toks[i] < (int)pal.size()) last = pal[toks[i]];
        std::copy(last.begin(), last.end(), img.rgb.begin() + i * 3);
    }
    return img;
}

Image sample_image(Model& m, float temperature, Rng& rng) {
    int size = m.spec.img_size;
    if (m.spec.kind != MODEL_GPT || m.spec.media != MEDIA_IMAGE || size <= 0) return Image();
    std::vector<int> toks = m.generate({0}, size * size, temperature, rng);
    return image_from_tokens(m.vocab, toks.data(), toks.size(), size);
}

Sound sample_sound(Model& m, float seconds, float temperature, Rng& rng, const std::function<bool(float)>& progress) {
    Sound snd;
    if (m.spec.kind != MODEL_GPT || m.spec.media != MEDIA_AUDIO || m.spec.rate <= 0) return snd;
    snd.rate = m.spec.rate;
    int n = std::max(1, (int)(seconds * snd.rate));
    int every = std::max(1, snd.rate / 20);
    snd.samples.reserve(n);
    m.generate({0}, n, temperature, rng, -1, [&](int t) {
        if (t == 0) {
            if ((int)snd.samples.size() > snd.rate / 2) return false;  // the model ended the clip
            snd.samples.push_back(0.f);
        } else {
            snd.samples.push_back(mulaw_decode(t - 1));
        }
        if (progress && snd.samples.size() % every == 0 && !progress((float)snd.samples.size() / n)) return false;
        return true;
    });
    return snd;
}

// ----------------------------------------------------------- checkpoints
//
// File layout (little endian):
//   "KOR1" | spec text | vocab[] | feat_names[] | norm_mean[] | norm_std[]
//   | u32 n_tensors | { name, i32 rows, i32 cols, u8 dtype(0=f32,1=f16), data }
//   [ "META" | u32 n | { key, value } ]   optional; older readers stop before it

namespace {
void wu32(std::ostream& o, uint32_t v) { o.write((const char*)&v, 4); }
void wstr(std::ostream& o, const std::string& s) { wu32(o, (uint32_t)s.size()); o.write(s.data(), (std::streamsize)s.size()); }
bool ru32(std::istream& i, uint32_t& v) { return (bool)i.read((char*)&v, 4); }
bool rstr(std::istream& i, std::string& s) {
    uint32_t n;
    if (!ru32(i, n) || n > (1u << 28)) return false;
    s.resize(n);
    return n == 0 || (bool)i.read(&s[0], n);
}
void wstrs(std::ostream& o, const std::vector<std::string>& v) { wu32(o, (uint32_t)v.size()); for (auto& s : v) wstr(o, s); }
bool rstrs(std::istream& i, std::vector<std::string>& v) {
    uint32_t n;
    if (!ru32(i, n) || n > (1u << 24)) return false;
    v.resize(n);
    for (auto& s : v) if (!rstr(i, s)) return false;
    return true;
}
void wfloats(std::ostream& o, const std::vector<float>& v) { wu32(o, (uint32_t)v.size()); o.write((const char*)v.data(), (std::streamsize)(v.size() * 4)); }
bool rfloats(std::istream& i, std::vector<float>& v) {
    uint32_t n;
    if (!ru32(i, n) || n > (1u << 24)) return false;
    v.resize(n);
    return n == 0 || (bool)i.read((char*)v.data(), (std::streamsize)(n * 4));
}
}  // namespace

bool Model::save(const std::string& path, bool half, std::string* err) {
    std::ofstream o(upath(path), std::ios::binary);
    if (!o) { if (err) *err = "cannot open file for writing"; return false; }
    o.write("KOR1", 4);
    wstr(o, spec.to_text());
    wstrs(o, vocab);
    wstrs(o, feat_names);
    wfloats(o, norm_mean);
    wfloats(o, norm_std);
    auto ps = params();
    wu32(o, (uint32_t)ps.size());
    std::vector<uint16_t> hbuf;
    for (Param* p : ps) {
        wstr(o, p->name);
        int32_t r = p->w.rows, c = p->w.cols;
        o.write((const char*)&r, 4);
        o.write((const char*)&c, 4);
        char dtype = half ? 1 : 0;
        o.write(&dtype, 1);
        if (half) {
            hbuf.resize(p->w.size());
            for (size_t i = 0; i < hbuf.size(); ++i) hbuf[i] = float_to_half(p->w.v[i]);
            o.write((const char*)hbuf.data(), (std::streamsize)(hbuf.size() * 2));
        } else {
            o.write((const char*)p->w.data(), (std::streamsize)(p->w.size() * 4));
        }
    }
    if (!meta.empty()) {
        o.write("META", 4);
        wu32(o, (uint32_t)meta.size());
        for (auto& kv : meta) { wstr(o, kv.first); wstr(o, kv.second); }
    }
    if (!o) { if (err) *err = "write failed"; return false; }
    return true;
}

std::shared_ptr<Model> Model::load(const std::string& path, std::string* err) {
    auto fail = [&](const char* m) { if (err) *err = m; return std::shared_ptr<Model>(); };
    std::ifstream in(upath(path), std::ios::binary);
    if (!in) return fail("cannot open file");
    char magic[4];
    if (!in.read(magic, 4) || std::memcmp(magic, "KOR1", 4) != 0) return fail("not a Kor model file");
    std::string text;
    ModelSpec spec;
    if (!rstr(in, text) || !spec.from_text(text)) return fail("corrupt model description");
    if (estimate_params(spec) > (size_t)2000000000) return fail("model too large");
    std::shared_ptr<Model> m(new Model(spec));
    if (!rstrs(in, m->vocab) || !rstrs(in, m->feat_names) || !rfloats(in, m->norm_mean) || !rfloats(in, m->norm_std))
        return fail("corrupt metadata");
    uint32_t n;
    auto ps = m->params();
    if (!ru32(in, n) || n != ps.size()) return fail("tensor count mismatch");
    std::vector<uint16_t> hbuf;
    for (Param* p : ps) {
        std::string name;
        int32_t r, c;
        char dtype;
        if (!rstr(in, name) || !in.read((char*)&r, 4) || !in.read((char*)&c, 4) || !in.read(&dtype, 1))
            return fail("corrupt tensor header");
        if (name != p->name || r != p->w.rows || c != p->w.cols) return fail("tensor shape mismatch");
        if (dtype == 1) {
            hbuf.resize(p->w.size());
            if (!in.read((char*)hbuf.data(), (std::streamsize)(hbuf.size() * 2))) return fail("truncated file");
            for (size_t i = 0; i < hbuf.size(); ++i) p->w.v[i] = half_to_float(hbuf[i]);
        } else {
            if (!in.read((char*)p->w.data(), (std::streamsize)(p->w.size() * 4))) return fail("truncated file");
        }
    }
    char tail[4];
    uint32_t nm = 0;
    if (in.read(tail, 4) && std::memcmp(tail, "META", 4) == 0 && ru32(in, nm) && nm < 10000)
        for (uint32_t k = 0; k < nm; ++k) {
            std::string key, value;
            if (!rstr(in, key) || !rstr(in, value)) break;
            m->meta[key] = value;
        }
    return m;
}

}  // namespace kor
