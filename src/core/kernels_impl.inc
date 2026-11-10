// Shared kernel bodies. Compiled once per instruction set: the including file
// defines KOR_KNS (namespace name) and is built with the matching flags, so
// the compiler auto-vectorizes the same loops for each target.
#include "kernels.h"
#include "parallel.h"

#include <cstddef>

#if defined(_MSC_VER)
#define KOR_RESTRICT __restrict
#else
#define KOR_RESTRICT __restrict__
#endif

namespace kor {
namespace KOR_KNS {

static inline float dot_i(const float* KOR_RESTRICT a, const float* KOR_RESTRICT b, int n) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int i = 0;
    for (; i + 8 <= n; i += 8)
        for (int j = 0; j < 8; ++j) acc[j] += a[i + j] * b[i + j];
    float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

static inline void axpy_i(float a, const float* KOR_RESTRICT x, float* KOR_RESTRICT y, int n) {
    for (int i = 0; i < n; ++i) y[i] += a * x[i];
}

float dot(const float* a, const float* b, int n) { return dot_i(a, b, n); }
void axpy(float a, const float* x, float* y, int n) { axpy_i(a, x, y, n); }

template <class F>
static inline void run(int n, double work, const F& body) {
    if (work < 60000.0) body(0, n);
    else parallel_for(n, body);
}

void mm(const float* A, const float* B, float* C, int m, int k, int n) {
    run(m, (double)m * k * n, [=](int i0, int i1) {
        for (int i = i0; i < i1; ++i) {
            const float* a = A + (size_t)i * k;
            float* c = C + (size_t)i * n;
            for (int p = 0; p < k; ++p) {
                float av = a[p];
                if (av == 0.f) continue;  // free win after ReLU
                axpy_i(av, B + (size_t)p * n, c, n);
            }
        }
    });
}

void mm_ta(const float* A, const float* B, float* C, int m, int k, int n) {
    run(k, (double)m * k * n, [=](int p0, int p1) {
        for (int p = p0; p < p1; ++p) {
            float* c = C + (size_t)p * n;
            for (int i = 0; i < m; ++i) {
                float av = A[(size_t)i * k + p];
                if (av == 0.f) continue;
                axpy_i(av, B + (size_t)i * n, c, n);
            }
        }
    });
}

void mm_tb(const float* A, const float* B, float* C, int m, int n, int k) {
    run(m, (double)m * k * n, [=](int i0, int i1) {
        for (int i = i0; i < i1; ++i) {
            const float* a = A + (size_t)i * n;
            float* c = C + (size_t)i * k;
            for (int p = 0; p < k; ++p) c[p] += dot_i(a, B + (size_t)p * n, n);
        }
    });
}

extern const Kernels table;
const Kernels table = {KOR_KNAME, mm, mm_ta, mm_tb, dot, axpy};

}  // namespace KOR_KNS
}  // namespace kor
