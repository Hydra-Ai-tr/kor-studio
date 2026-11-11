// Unpacking of GGUF weight blocks, built once per instruction set (see
// gguf_dequant_base.cpp and gguf_dequant_avx2.cpp); KOR_GNS names the copy.
#include <cstdint>
#include <cstring>
#include "gguf.h"

namespace kor {
namespace KOR_GNS {
namespace {

float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, f;
    if (e == 0) {
        if (m == 0) {
            f = s;
        } else {
            e = 113;
            while (!(m & 0x400)) { m <<= 1; --e; }
            f = s | (e << 23) | ((m & 0x3FF) << 13);
        }
    } else if (e == 31) {
        f = s | 0x7F800000 | (m << 13);
    } else {
        f = s | ((e + 112) << 23) | (m << 13);
    }
    float r;
    std::memcpy(&r, &f, 4);
    return r;
}

float bf2f(uint16_t b) {
    uint32_t f = (uint32_t)b << 16;
    float r;
    std::memcpy(&r, &f, 4);
    return r;
}

uint16_t rd16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline void scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

}  // namespace

// Unpacks nb whole blocks (the reference formulas of ggml).
void dequant_blocks(int type, const uint8_t* x, int64_t nb, float* y) {
    switch (type) {
    case GGML_F32:
        std::memcpy(y, x, (size_t)nb * 4);
        return;
    case GGML_F16:
        for (int64_t i = 0; i < nb; ++i) y[i] = h2f(rd16(x + 2 * i));
        return;
    case GGML_BF16:
        for (int64_t i = 0; i < nb; ++i) y[i] = bf2f(rd16(x + 2 * i));
        return;
    case GGML_Q4_0:
        for (int64_t i = 0; i < nb; ++i, x += 18, y += 32) {
            float d = h2f(rd16(x));
            const uint8_t* q = x + 2;
            for (int j = 0; j < 16; ++j) {
                y[j] = ((q[j] & 0xF) - 8) * d;
                y[j + 16] = ((q[j] >> 4) - 8) * d;
            }
        }
        return;
    case GGML_Q4_1:
        for (int64_t i = 0; i < nb; ++i, x += 20, y += 32) {
            float d = h2f(rd16(x)), m = h2f(rd16(x + 2));
            const uint8_t* q = x + 4;
            for (int j = 0; j < 16; ++j) {
                y[j] = (q[j] & 0xF) * d + m;
                y[j + 16] = (q[j] >> 4) * d + m;
            }
        }
        return;
    case GGML_Q5_0:
        for (int64_t i = 0; i < nb; ++i, x += 22, y += 32) {
            float d = h2f(rd16(x));
            uint32_t qh;
            std::memcpy(&qh, x + 2, 4);
            const uint8_t* q = x + 6;
            for (int j = 0; j < 16; ++j) {
                int h0 = ((qh >> j) << 4) & 0x10, h1 = (qh >> (j + 12)) & 0x10;
                y[j] = (((q[j] & 0xF) | h0) - 16) * d;
                y[j + 16] = (((q[j] >> 4) | h1) - 16) * d;
            }
        }
        return;
    case GGML_Q5_1:
        for (int64_t i = 0; i < nb; ++i, x += 24, y += 32) {
            float d = h2f(rd16(x)), m = h2f(rd16(x + 2));
            uint32_t qh;
            std::memcpy(&qh, x + 4, 4);
            const uint8_t* q = x + 8;
            for (int j = 0; j < 16; ++j) {
                int h0 = ((qh >> j) << 4) & 0x10, h1 = (qh >> (j + 12)) & 0x10;
                y[j] = ((q[j] & 0xF) | h0) * d + m;
                y[j + 16] = ((q[j] >> 4) | h1) * d + m;
            }
        }
        return;
    case GGML_Q8_0:
        for (int64_t i = 0; i < nb; ++i, x += 34, y += 32) {
            float d = h2f(rd16(x));
            const int8_t* q = (const int8_t*)(x + 2);
            for (int j = 0; j < 32; ++j) y[j] = q[j] * d;
        }
        return;
    case GGML_Q2_K:
        for (int64_t i = 0; i < nb; ++i, x += 84) {
            const uint8_t* sc = x;
            const uint8_t* q = x + 16;
            float d = h2f(rd16(x + 80)), mn = h2f(rd16(x + 82));
            int is = 0;
            for (int n = 0; n < 256; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    uint8_t s = sc[is++];
                    float dl = d * (s & 0xF), ml = mn * (s >> 4);
                    for (int l = 0; l < 16; ++l) *y++ = dl * ((q[l] >> shift) & 3) - ml;
                    s = sc[is++];
                    dl = d * (s & 0xF);
                    ml = mn * (s >> 4);
                    for (int l = 0; l < 16; ++l) *y++ = dl * ((q[l + 16] >> shift) & 3) - ml;
                    shift += 2;
                }
                q += 32;
            }
        }
        return;
    case GGML_Q3_K:
        for (int64_t i = 0; i < nb; ++i, x += 110) {
            const uint8_t* hm = x;
            const uint8_t* q = x + 32;
            float dall = h2f(rd16(x + 108));
            const uint32_t km1 = 0x03030303, km2 = 0x0f0f0f0f;
            uint32_t aux[4];
            std::memcpy(aux, x + 96, 12);
            uint32_t tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & km2) | (((tmp >> 4) & km1) << 4);
            aux[3] = ((aux[1] >> 4) & km2) | (((tmp >> 6) & km1) << 4);
            aux[0] = (aux[0] & km2) | (((tmp >> 0) & km1) << 4);
            aux[1] = (aux[1] & km2) | (((tmp >> 2) & km1) << 4);
            int8_t scales[16];
            std::memcpy(scales, aux, 16);
            int is = 0;
            uint8_t m = 1;
            for (int n = 0; n < 256; n += 128) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    float dl = dall * (scales[is++] - 32);
                    for (int l = 0; l < 16; ++l) *y++ = dl * ((int)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                    dl = dall * (scales[is++] - 32);
                    for (int l = 0; l < 16; ++l) *y++ = dl * ((int)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                    shift += 2;
                    m <<= 1;
                }
                q += 32;
            }
        }
        return;
    case GGML_Q4_K:
        for (int64_t i = 0; i < nb; ++i, x += 144) {
            float d = h2f(rd16(x)), mn = h2f(rd16(x + 2));
            const uint8_t* sc = x + 4;
            const uint8_t* q = x + 16;
            int is = 0;
            uint8_t s, m;
            for (int j = 0; j < 256; j += 64) {
                scale_min_k4(is, sc, &s, &m);
                float d1 = d * s, m1 = mn * m;
                scale_min_k4(is + 1, sc, &s, &m);
                float d2 = d * s, m2 = mn * m;
                for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
                for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
                q += 32;
                is += 2;
            }
        }
        return;
    case GGML_Q5_K:
        for (int64_t i = 0; i < nb; ++i, x += 176) {
            float d = h2f(rd16(x)), mn = h2f(rd16(x + 2));
            const uint8_t* sc = x + 4;
            const uint8_t* qh = x + 16;
            const uint8_t* ql = x + 48;
            int is = 0;
            uint8_t s, m, u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                scale_min_k4(is, sc, &s, &m);
                float d1 = d * s, m1 = mn * m;
                scale_min_k4(is + 1, sc, &s, &m);
                float d2 = d * s, m2 = mn * m;
                for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
                for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
                ql += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        return;
    case GGML_Q6_K:
        for (int64_t i = 0; i < nb; ++i, x += 210) {
            const uint8_t* ql = x;
            const uint8_t* qh = x + 128;
            const int8_t* sc = (const int8_t*)(x + 192);
            float d = h2f(rd16(x + 208));
            for (int n = 0; n < 256; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l] = d * sc[is] * q1;
                    y[l + 32] = d * sc[is + 2] * q2;
                    y[l + 64] = d * sc[is + 4] * q3;
                    y[l + 96] = d * sc[is + 6] * q4;
                }
                y += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        return;
    default:
        std::memset(y, 0, sizeof(float) * (size_t)nb);
    }
}

}  // namespace KOR_GNS
}  // namespace kor

