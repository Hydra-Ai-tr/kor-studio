#include "layers.h"

#include "kernels.h"
#include "parallel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>

namespace kor {

// ---------------------------------------------------------------- Linear

Linear::Linear(int in_, int out_, const std::string& name, Rng& rng, float stddev)
    : in(in_), out(out_) {
    W.init(name + ".W", in, out, true);
    b.init(name + ".b", 1, out, false);
    if (stddev < 0) stddev = std::sqrt(2.f / (float)in);  // He init
    for (auto& x : W.w.v) x = rng.normal() * stddev;
}

void Linear::enable_lora(int r, Rng& rng) {
    if (r <= 0 || rank > 0) return;
    rank = r;
    lscale = 2.f;  // alpha / rank with alpha = 2 * rank
    A.init(W.name + ".loraA", in, r, false);
    Bm.init(W.name + ".loraB", r, out, false);
    float sd = 1.f / std::sqrt((float)in);
    for (auto& x : A.w.v) x = rng.normal() * sd;  // B starts at zero: no change at step 0
    W.freeze();
    b.freeze();
}

void Linear::merge_lora() {
    if (rank <= 0) return;
    Tensor delta(in, out);
    K().mm(A.w.data(), Bm.w.data(), delta.data(), in, rank, out);
    for (size_t i = 0; i < delta.size(); ++i) W.w.v[i] += lscale * delta.v[i];
    rank = 0;
    A = Param(); Bm = Param();
    W.trainable = true; W.g.resize(in, out);
    b.trainable = true; b.g.resize(1, out);
}

void Linear::forward(const Tensor& x, Tensor& y, bool) {
    xin = &x;
    int n = x.rows;
    y.resize_noinit(n, out);
    for (int r = 0; r < n; ++r) std::memcpy(y.row(r), b.w.data(), sizeof(float) * out);
    K().mm(x.data(), W.w.data(), y.data(), n, in, out);
    if (rank > 0) {
        xa.resize(n, rank);
        K().mm(x.data(), A.w.data(), xa.data(), n, in, rank);
        for (auto& t : xa.v) t *= lscale;
        K().mm(xa.data(), Bm.w.data(), y.data(), n, rank, out);
    }
}

void Linear::backward(const Tensor& dy, Tensor& dx) {
    const Tensor& x = *xin;
    int n = x.rows;
    if (W.trainable) K().mm_ta(x.data(), dy.data(), W.g.data(), n, in, out);
    if (b.trainable) {
        float* bg = b.g.data();
        for (int r = 0; r < n; ++r) {
            const float* d = dy.row(r);
            for (int j = 0; j < out; ++j) bg[j] += d[j];
        }
    }
    if (need_dx) {
        dx.resize(n, in);
        K().mm_tb(dy.data(), W.w.data(), dx.data(), n, out, in);
    }
    if (rank > 0) {
        K().mm_ta(xa.data(), dy.data(), Bm.g.data(), n, rank, out);
        dxa.resize(n, rank);
        K().mm_tb(dy.data(), Bm.w.data(), dxa.data(), n, out, rank);
        for (auto& t : dxa.v) t *= lscale;
        K().mm_ta(x.data(), dxa.data(), A.g.data(), n, in, rank);
        if (need_dx) K().mm_tb(dxa.data(), A.w.data(), dx.data(), n, rank, in);
    }
}

void Linear::collect(std::vector<Param*>& ps) {
    ps.push_back(&W);
    ps.push_back(&b);
    if (rank > 0) {
        ps.push_back(&A);
        ps.push_back(&Bm);
    }
}

// ------------------------------------------------------------ Activation

static const float kGeluC = 0.7978845608f;  // sqrt(2/pi)

void Activation::forward(const Tensor& x, Tensor& y, bool) {
    xin = &x; yout = &y;
    y.resize_noinit(x.rows, x.cols);
    const float* a = x.data();
    float* o = y.data();
    size_t n = x.size();
    switch (kind) {
        case ACT_RELU:
            for (size_t i = 0; i < n; ++i) o[i] = a[i] > 0 ? a[i] : 0.f;
            break;
        case ACT_GELU:
            for (size_t i = 0; i < n; ++i) {
                float v = a[i];
                o[i] = 0.5f * v * (1.f + std::tanh(kGeluC * (v + 0.044715f * v * v * v)));
            }
            break;
        case ACT_TANH:
            for (size_t i = 0; i < n; ++i) o[i] = std::tanh(a[i]);
            break;
        default:
            for (size_t i = 0; i < n; ++i) o[i] = 1.f / (1.f + std::exp(-a[i]));
            break;
    }
}

void Activation::backward(const Tensor& dy, Tensor& dx) {
    const float* a = xin->data();
    const float* o = yout->data();
    const float* d = dy.data();
    dx.resize_noinit(dy.rows, dy.cols);
    float* r = dx.data();
    size_t n = dy.size();
    switch (kind) {
        case ACT_RELU:
            for (size_t i = 0; i < n; ++i) r[i] = o[i] > 0 ? d[i] : 0.f;
            break;
        case ACT_GELU:
            for (size_t i = 0; i < n; ++i) {
                float v = a[i];
                float t = std::tanh(kGeluC * (v + 0.044715f * v * v * v));
                float dt = (1.f - t * t) * kGeluC * (1.f + 3.f * 0.044715f * v * v);
                r[i] = d[i] * (0.5f * (1.f + t) + 0.5f * v * dt);
            }
            break;
        case ACT_TANH:
            for (size_t i = 0; i < n; ++i) r[i] = d[i] * (1.f - o[i] * o[i]);
            break;
        default:
            for (size_t i = 0; i < n; ++i) r[i] = d[i] * o[i] * (1.f - o[i]);
            break;
    }
}

// --------------------------------------------------------------- Dropout

void Dropout::forward(const Tensor& x, Tensor& y, bool train) {
    y.resize_noinit(x.rows, x.cols);
    active = train && p > 0.f;
    size_t n = x.size();
    if (!active) {
        std::memcpy(y.data(), x.data(), sizeof(float) * n);
        return;
    }
    mask.resize(n);
    float keep = 1.f / (1.f - p);
    for (size_t i = 0; i < n; ++i) {
        mask[i] = rng.uniform() >= p;
        y.v[i] = mask[i] ? x.v[i] * keep : 0.f;
    }
}

void Dropout::backward(const Tensor& dy, Tensor& dx) {
    dx.resize_noinit(dy.rows, dy.cols);
    size_t n = dy.size();
    if (!active) {
        std::memcpy(dx.data(), dy.data(), sizeof(float) * n);
        return;
    }
    float keep = 1.f / (1.f - p);
    for (size_t i = 0; i < n; ++i) dx.v[i] = mask[i] ? dy.v[i] * keep : 0.f;
}

// ------------------------------------------------------------- LayerNorm

LayerNorm::LayerNorm(int dim_, const std::string& name, bool rms_) : dim(dim_), rms(rms_) {
    g.init(name + ".g", 1, dim, false);
    if (!rms) b.init(name + ".b", 1, dim, false);
    for (auto& x : g.w.v) x = 1.f;
}

void LayerNorm::forward(const Tensor& x, Tensor& y, bool) {
    int n = x.rows, d = dim;
    y.resize_noinit(n, d);
    xhat.resize_noinit(n, d);
    inv.resize(n);
    const float* gw = g.w.data();
    if (rms) {
        for (int r = 0; r < n; ++r) {
            const float* a = x.row(r);
            float ss = 0;
            for (int j = 0; j < d; ++j) ss += a[j] * a[j];
            float is = 1.f / std::sqrt(ss / d + 1e-5f);
            inv[r] = is;
            float* xh = xhat.row(r);
            float* o = y.row(r);
            for (int j = 0; j < d; ++j) {
                xh[j] = a[j] * is;
                o[j] = gw[j] * xh[j];
            }
        }
        return;
    }
    const float* bw = b.w.data();
    for (int r = 0; r < n; ++r) {
        const float* a = x.row(r);
        float mean = 0;
        for (int j = 0; j < d; ++j) mean += a[j];
        mean /= d;
        float var = 0;
        for (int j = 0; j < d; ++j) var += (a[j] - mean) * (a[j] - mean);
        float is = 1.f / std::sqrt(var / d + 1e-5f);
        inv[r] = is;
        float* xh = xhat.row(r);
        float* o = y.row(r);
        for (int j = 0; j < d; ++j) {
            xh[j] = (a[j] - mean) * is;
            o[j] = gw[j] * xh[j] + bw[j];
        }
    }
}

void LayerNorm::backward(const Tensor& dy, Tensor& dx) {
    int n = dy.rows, d = dim;
    dx.resize_noinit(n, d);
    const float* gw = g.w.data();
    bool tr = g.trainable;
    float* gg = tr ? g.g.data() : nullptr;
    if (rms) {
        // y = g * xh with xh = x * is: dx = is * (dy * g - xh * mean(dy * g * xh))
        for (int r = 0; r < n; ++r) {
            const float* dyr = dy.row(r);
            const float* xh = xhat.row(r);
            float m2 = 0;
            for (int j = 0; j < d; ++j) m2 += dyr[j] * gw[j] * xh[j];
            if (tr)
                for (int j = 0; j < d; ++j) gg[j] += dyr[j] * xh[j];
            m2 /= d;
            float is = inv[r];
            float* o = dx.row(r);
            for (int j = 0; j < d; ++j) o[j] = is * (dyr[j] * gw[j] - xh[j] * m2);
        }
        return;
    }
    float* bg = tr ? b.g.data() : nullptr;
    for (int r = 0; r < n; ++r) {
        const float* dyr = dy.row(r);
        const float* xh = xhat.row(r);
        float m1 = 0, m2 = 0;
        for (int j = 0; j < d; ++j) {
            float dxh = dyr[j] * gw[j];
            m1 += dxh;
            m2 += dxh * xh[j];
        }
        if (tr)
            for (int j = 0; j < d; ++j) {
                gg[j] += dyr[j] * xh[j];
                bg[j] += dyr[j];
            }
        m1 /= d; m2 /= d;
        float is = inv[r];
        float* o = dx.row(r);
        for (int j = 0; j < d; ++j) o[j] = is * (dyr[j] * gw[j] - m1 - xh[j] * m2);
    }
}

void LayerNorm::collect(std::vector<Param*>& ps) {
    ps.push_back(&g);
    if (!rms) ps.push_back(&b);
}

// ------------------------------------------------------------- Embedding

Embedding::Embedding(int vocab_, int ctx_, int dim_, Rng& rng, bool learned_pos_)
    : vocab(vocab_), ctx(ctx_), dim(dim_), learned_pos(learned_pos_) {
    tok.init("emb.tok", vocab, dim, false);
    for (auto& x : tok.w.v) x = rng.normal() * 0.02f;
    if (learned_pos) {
        pos.init("emb.pos", ctx, dim, false);
        for (auto& x : pos.w.v) x = rng.normal() * 0.02f;
    }
}

void Embedding::forward(const std::vector<int>& ids, int B_, int T_, Tensor& y) {
    ids_in = &ids; B = B_; T = T_;
    y.resize_noinit(B * T, dim);
    for (int n = 0; n < B * T; ++n) {
        const float* a = tok.w.row(ids[n]);
        float* o = y.row(n);
        if (!learned_pos) {
            std::memcpy(o, a, sizeof(float) * dim);
            continue;
        }
        const float* p = pos.w.row(n % T);
        for (int j = 0; j < dim; ++j) o[j] = a[j] + p[j];
    }
}

void Embedding::backward(const Tensor& dy) {
    if (!tok.trainable) return;
    const std::vector<int>& ids = *ids_in;
    for (int n = 0; n < B * T; ++n) {
        const float* d = dy.row(n);
        float* a = tok.g.row(ids[n]);
        for (int j = 0; j < dim; ++j) a[j] += d[j];
        if (!learned_pos) continue;
        float* p = pos.g.row(n % T);
        for (int j = 0; j < dim; ++j) p[j] += d[j];
    }
}

void Embedding::collect(std::vector<Param*>& ps) {
    ps.push_back(&tok);
    if (learned_pos) ps.push_back(&pos);
}

// ------------------------------------------------------------- Attention

Attention::Attention(int dim, int heads, const std::string& name, Rng& rng, float proj_std, float theta)
    : C(dim), H(heads), hd(dim / heads), rope_theta(theta),
      qkv(dim, 3 * dim, name + ".qkv", rng, 0.02f),
      proj(dim, dim, name + ".proj", rng, proj_std) {}

void Attention::rope(Tensor& m, int n, int T_, int pos, bool back) {
    const int half = hd / 2;
    int npos = pos >= 0 ? pos + 1 : std::min(n, T_);
    if ((int)(rcos.size() / std::max(1, half)) < npos) {
        int have = (int)(rcos.size() / std::max(1, half));
        int want = std::max(npos, 2 * have);
        rcos.resize((size_t)want * half);
        rsin.resize((size_t)want * half);
        for (int p = have; p < want; ++p)
            for (int i = 0; i < half; ++i) {
                double a = p * std::pow((double)rope_theta, -2.0 * i / hd);
                rcos[(size_t)p * half + i] = (float)std::cos(a);
                rsin[(size_t)p * half + i] = (float)std::sin(a);
            }
    }
    const float sg = back ? -1.f : 1.f;
    parallel_for(n, [&](int s, int e) {
        for (int r = s; r < e; ++r) {
            int p = pos >= 0 ? pos : r % T_;
            const float* c = rcos.data() + (size_t)p * half;
            const float* sn = rsin.data() + (size_t)p * half;
            float* row = m.row(r);
            for (int part = 0; part < 2; ++part)  // query, key
                for (int h = 0; h < H; ++h) {
                    float* v = row + part * C + h * hd;
                    for (int i = 0; i < half; ++i) {
                        float x1 = v[i], x2 = v[i + half], si = sg * sn[i];
                        v[i] = x1 * c[i] - x2 * si;
                        v[i + half] = x1 * si + x2 * c[i];
                    }
                }
        }
    });
}

void Attention::forward(const Tensor& x, Tensor& y, int B_, int T_) {
    B = B_; T = T_;
    qkv.forward(x, q3, true);
    if (rope_theta > 0) rope(q3, B * T, T, -1, false);
    const Kernels& k = K();
    ho.resize_noinit(B * T, C);
    on_gpu = k.attn_fwd && k.attn_fwd(q3.data(), ho.data(), B, T, H, hd);
    if (!on_gpu) cpu_forward();
    proj.forward(ho, y, true);
}

void Attention::cpu_forward() {
    ho.resize(B * T, C);
    att.assign((size_t)B * H * T * T, 0.f);
    const Kernels& k = K();
    float scale = 1.f / std::sqrt((float)hd);
    parallel_for(B * H, [&](int s, int e) {
        for (int idx = s; idx < e; ++idx) {
            int b = idx / H, h = idx % H;
            float* a = att.data() + (size_t)idx * T * T;
            for (int i = 0; i < T; ++i) {
                const float* qi = q3.row(b * T + i) + h * hd;
                float* ai = a + (size_t)i * T;
                float mx = -1e30f;
                for (int j = 0; j <= i; ++j) {
                    ai[j] = k.dot(qi, q3.row(b * T + j) + C + h * hd, hd) * scale;
                    if (ai[j] > mx) mx = ai[j];
                }
                float sum = 0;
                for (int j = 0; j <= i; ++j) {
                    ai[j] = std::exp(ai[j] - mx);
                    sum += ai[j];
                }
                float is = 1.f / sum;
                float* o = ho.row(b * T + i) + h * hd;
                for (int j = 0; j <= i; ++j) {
                    ai[j] *= is;
                    k.axpy(ai[j], q3.row(b * T + j) + 2 * C + h * hd, o, hd);
                }
            }
        }
    });
}

void Attention::step(const Tensor& x, Tensor& y, Tensor& kv, int pos) {
    qkv.forward(x, q3, false);
    if (rope_theta > 0) rope(q3, 1, 1, pos, false);
    const float* q = q3.row(0);
    std::memcpy(kv.row(pos), q + C, sizeof(float) * 2 * C);
    ho.resize(1, C);
    att.resize((size_t)pos + 1);
    const Kernels& k = K();
    float scale = 1.f / std::sqrt((float)hd);
    for (int h = 0; h < H; ++h) {
        const float* qh = q + h * hd;
        float mx = -1e30f;
        for (int j = 0; j <= pos; ++j) {
            att[j] = k.dot(qh, kv.row(j) + h * hd, hd) * scale;
            mx = std::max(mx, att[j]);
        }
        float sum = 0;
        for (int j = 0; j <= pos; ++j) {
            att[j] = std::exp(att[j] - mx);
            sum += att[j];
        }
        float* o = ho.row(0) + h * hd;
        for (int j = 0; j <= pos; ++j) k.axpy(att[j] / sum, kv.row(j) + C + h * hd, o, hd);
    }
    proj.forward(ho, y, false);
}

void Attention::backward(const Tensor& dy, Tensor& dx) {
    proj.backward(dy, dho);
    const Kernels& k = K();
    if (on_gpu && k.attn_bwd) {
        dq3.resize_noinit(B * T, 3 * C);
        if (k.attn_bwd(q3.data(), dho.data(), dq3.data(), B, T, H, hd)) {
            if (rope_theta > 0) rope(dq3, B * T, T, -1, true);
            qkv.backward(dq3, dx);
            return;
        }
    }
    if (on_gpu) {  // the GPU kept no attention weights: recompute them
        cpu_forward();
        on_gpu = false;
    }
    dq3.resize(B * T, 3 * C);
    float scale = 1.f / std::sqrt((float)hd);
    parallel_for(B * H, [&](int s, int e) {
        std::vector<float> datt(T);
        for (int idx = s; idx < e; ++idx) {
            int b = idx / H, h = idx % H;
            const float* a = att.data() + (size_t)idx * T * T;
            for (int i = 0; i < T; ++i) {
                const float* doi = dho.row(b * T + i) + h * hd;
                const float* qi = q3.row(b * T + i) + h * hd;
                float* dqi = dq3.row(b * T + i) + h * hd;
                const float* ai = a + (size_t)i * T;
                float dsum = 0;
                for (int j = 0; j <= i; ++j) {
                    datt[j] = k.dot(doi, q3.row(b * T + j) + 2 * C + h * hd, hd);
                    dsum += ai[j] * datt[j];
                    k.axpy(ai[j], doi, dq3.row(b * T + j) + 2 * C + h * hd, hd);  // dV
                }
                for (int j = 0; j <= i; ++j) {
                    float ds = ai[j] * (datt[j] - dsum) * scale;
                    if (ds == 0.f) continue;
                    k.axpy(ds, q3.row(b * T + j) + C + h * hd, dqi, hd);          // dQ
                    k.axpy(ds, qi, dq3.row(b * T + j) + C + h * hd, hd);          // dK
                }
            }
        }
    });
    if (rope_theta > 0) rope(dq3, B * T, T, -1, true);
    qkv.backward(dq3, dx);
}

// ----------------------------------------------------------------- Block

Block::Block(int dim, int heads, const std::string& name, Rng& rng, float resid_std, int ffn, float rope_theta)
    : modern(ffn > 0),
      ln1(dim, name + ".ln1", ffn > 0), ln2(dim, name + ".ln2", ffn > 0),
      attn(dim, heads, name + ".attn", rng, resid_std, ffn > 0 ? rope_theta : 0.f),
      fc1(dim, ffn > 0 ? 2 * ffn : 4 * dim, name + ".fc1", rng, 0.02f),
      fc2(ffn > 0 ? ffn : 4 * dim, dim, name + ".fc2", rng, resid_std),
      act(ACT_GELU),
      drop1(0.f, std::hash<std::string>()(name + ".drop1")),
      drop2(0.f, std::hash<std::string>()(name + ".drop2")) {}

void Block::set_dropout(float p) { drop1.p = drop2.p = p; }

namespace {

// SwiGLU: a rows are [gate | up] (2F), y = silu(gate) * up (F).
void swiglu(const Tensor& a, Tensor& y) {
    int n = a.rows, F = a.cols / 2;
    y.resize_noinit(n, F);
    parallel_for(n, [&](int s, int e) {
        for (int r = s; r < e; ++r) {
            const float* g = a.row(r);
            const float* u = g + F;
            float* o = y.row(r);
            for (int j = 0; j < F; ++j) o[j] = g[j] / (1.f + std::exp(-g[j])) * u[j];
        }
    });
}

void swiglu_back(const Tensor& a, const Tensor& dy, Tensor& da) {
    int n = a.rows, F = a.cols / 2;
    da.resize_noinit(n, 2 * F);
    parallel_for(n, [&](int s, int e) {
        for (int r = s; r < e; ++r) {
            const float* g = a.row(r);
            const float* u = g + F;
            const float* d = dy.row(r);
            float* o = da.row(r);
            for (int j = 0; j < F; ++j) {
                float sg = 1.f / (1.f + std::exp(-g[j])), si = g[j] * sg;
                o[j] = d[j] * u[j] * sg * (1.f + g[j] * (1.f - sg));
                o[j + F] = d[j] * si;
            }
        }
    });
}

}  // namespace

void Block::forward(const Tensor& x, Tensor& y, int B, int T, bool train) {
    ln1.forward(x, a1, true);
    attn.forward(a1, a2, B, T);
    drop1.forward(a2, r1, train);
    h.resize_noinit(x.rows, x.cols);
    for (size_t i = 0; i < x.size(); ++i) h.v[i] = x.v[i] + r1.v[i];
    ln2.forward(h, a3, true);
    fc1.forward(a3, a4, true);
    if (modern) swiglu(a4, a5);
    else act.forward(a4, a5, true);
    fc2.forward(a5, a6, true);
    drop2.forward(a6, r2, train);
    y.resize_noinit(x.rows, x.cols);
    for (size_t i = 0; i < x.size(); ++i) y.v[i] = h.v[i] + r2.v[i];
}

void Block::step(const Tensor& x, Tensor& y, Tensor& kv, int pos) {
    ln1.forward(x, a1, false);
    attn.step(a1, a2, kv, pos);
    h.resize_noinit(1, x.cols);
    for (int i = 0; i < x.cols; ++i) h.v[i] = x.v[i] + a2.v[i];
    ln2.forward(h, a3, false);
    fc1.forward(a3, a4, false);
    if (modern) swiglu(a4, a5);
    else act.forward(a4, a5, false);
    fc2.forward(a5, a6, false);
    y.resize_noinit(1, x.cols);
    for (int i = 0; i < x.cols; ++i) y.v[i] = h.v[i] + a6.v[i];
}

void Block::backward(const Tensor& dy, Tensor& dx) {
    drop2.backward(dy, t2);
    fc2.backward(t2, t1);
    if (modern) swiglu_back(a4, t1, t2);
    else act.backward(t1, t2);
    fc1.backward(t2, t1);
    ln2.backward(t1, t2);
    dh.resize_noinit(dy.rows, dy.cols);
    for (size_t i = 0; i < dy.size(); ++i) dh.v[i] = dy.v[i] + t2.v[i];
    drop1.backward(dh, t2);
    attn.backward(t2, t1);
    ln1.backward(t1, t2);
    dx.resize_noinit(dy.rows, dy.cols);
    for (size_t i = 0; i < dy.size(); ++i) dx.v[i] = dh.v[i] + t2.v[i];
}

void Block::collect(std::vector<Param*>& ps) {
    ln1.collect(ps);
    attn.qkv.collect(ps);
    attn.proj.collect(ps);
    ln2.collect(ps);
    fc1.collect(ps);
    fc2.collect(ps);
}

void Block::linears(std::vector<Linear*>& ls) {
    ls.push_back(&attn.qkv);
    ls.push_back(&attn.proj);
    ls.push_back(&fc1);
    ls.push_back(&fc2);
}

// ---------------------------------------------------------------- Losses

float cross_entropy(const Tensor& logits, const int* target, Tensor* d, float scale, int* correct, const float* weight) {
    int n = logits.rows, c = logits.cols;
    if (d) d->resize_noinit(n, c);
    double total = 0, wsum = n;
    int ok = 0;
    if (weight) {
        wsum = 0;
        for (int r = 0; r < n; ++r) wsum += weight[r];
        if (wsum <= 0) wsum = 1;
    }
    float base = scale / (float)wsum;
    for (int r = 0; r < n; ++r) {
        float w = weight ? weight[r] : 1.f;
        float gs = base * w;
        const float* z = logits.row(r);
        float mx = z[0];
        int arg = 0;
        for (int j = 1; j < c; ++j)
            if (z[j] > mx) { mx = z[j]; arg = j; }
        double sum = 0;
        for (int j = 0; j < c; ++j) sum += std::exp((double)(z[j] - mx));
        int t = target[r];
        total += w * (std::log(sum) - (double)(z[t] - mx));
        if (arg == t) ++ok;
        if (d) {
            float* o = d->row(r);
            float is = (float)(1.0 / sum);
            for (int j = 0; j < c; ++j) o[j] = std::exp(z[j] - mx) * is * gs;
            o[t] -= gs;
        }
    }
    if (correct) *correct = ok;
    return (float)(total / wsum);
}

float mse_loss(const Tensor& pred, const Tensor& y, Tensor* d, float scale) {
    size_t n = pred.size();
    if (d) d->resize_noinit(pred.rows, pred.cols);
    double total = 0;
    float gs = 2.f * scale / (float)n;
    for (size_t i = 0; i < n; ++i) {
        float e = pred.v[i] - y.v[i];
        total += (double)e * e;
        if (d) d->v[i] = e * gs;
    }
    return (float)(total / n);
}

}  // namespace kor
