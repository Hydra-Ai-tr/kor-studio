// Kor Studio - minimal 2D float tensor, parameter and RNG types.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kor {

// Row-major 2D float matrix. Everything in the engine is expressed as
// [rows x cols]; sequence models flatten (batch, time) into rows.
struct Tensor {
    int rows = 0, cols = 0;
    std::vector<float> v;

    Tensor() = default;
    Tensor(int r, int c) : rows(r), cols(c), v((size_t)r * c, 0.f) {}

    void resize(int r, int c) {  // resizes and zero-fills
        rows = r; cols = c;
        v.assign((size_t)r * c, 0.f);
    }
    void resize_noinit(int r, int c) {  // contents unspecified
        rows = r; cols = c;
        v.resize((size_t)r * c);
    }
    void zero() { std::fill(v.begin(), v.end(), 0.f); }
    size_t size() const { return v.size(); }
    float* data() { return v.data(); }
    const float* data() const { return v.data(); }
    float* row(int r) { return v.data() + (size_t)r * cols; }
    const float* row(int r) const { return v.data() + (size_t)r * cols; }
};

// A learnable tensor with its gradient and optimizer state.
struct Param {
    std::string name;
    Tensor w, g;   // value, gradient
    Tensor m, s;   // optimizer moments (allocated lazily)
    bool trainable = true;
    bool decay = true;  // apply weight decay

    void init(const std::string& n, int r, int c, bool decay_) {
        name = n; decay = decay_;
        w.resize(r, c); g.resize(r, c);
    }
    void freeze() {
        trainable = false;
        g = Tensor(); m = Tensor(); s = Tensor();
    }
};

// xorshift64* generator: fast, deterministic across platforms.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed = 0x9E3779B97F4A7C15ull) : s(seed ? seed : 1) {}
    uint64_t next() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 0x2545F4914F6CDD1Dull;
    }
    float uniform() { return (float)((next() >> 40) * (1.0 / 16777216.0)); }  // [0,1)
    int below(int n) { return (int)(next() % (uint64_t)n); }
    float normal() {
        float u1 = uniform(), u2 = uniform();
        if (u1 < 1e-7f) u1 = 1e-7f;
        return std::sqrt(-2.f * std::log(u1)) * std::cos(6.2831853f * u2);
    }
};

}  // namespace kor
