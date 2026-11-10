#include "media.h"

#include "model.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace kor {

static bool read_all(const std::string& path, std::vector<uint8_t>& out, std::string* err) {
    std::ifstream in(upath(path), std::ios::binary);
    if (!in) { if (err) *err = "cannot open file"; return false; }
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string s = ss.str();
    out.assign(s.begin(), s.end());
    return true;
}

static std::string lower_ext(const std::string& path) {
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return "";
    std::string e = path.substr(dot + 1);
    for (auto& c : e) c = (char)std::tolower((unsigned char)c);
    return e;
}

bool is_image_file(const std::string& path) {
    std::string e = lower_ext(path);
    return e == "png" || e == "bmp" || e == "ppm" || e == "pgm";
}

bool is_wav_file(const std::string& path) { return lower_ext(path) == "wav"; }

// ------------------------------------------------------------------ crc32

uint32_t crc32(const uint8_t* data, size_t n, uint32_t crc) {
    static uint32_t table[256];
    static bool init = [] {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        return true;
    }();
    (void)init;
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xff] ^ (crc >> 8);
    return ~crc;
}

// ---------------------------------------------------------------- inflate

namespace {

struct Bits {
    const uint8_t* p;
    size_t n, pos = 0;
    uint32_t buf = 0;
    int cnt = 0;
    bool bad = false;
    uint32_t get(int need) {
        while (cnt < need) {
            if (pos >= n) { bad = true; return 0; }
            buf |= (uint32_t)p[pos++] << cnt;
            cnt += 8;
        }
        uint32_t v = buf & ((1u << need) - 1);
        buf >>= need;
        cnt -= need;
        return v;
    }
};

struct Huff {
    short count[16] = {0};
    std::vector<short> symbol;
};

bool build(Huff& h, const short* len, int n) {
    std::fill(h.count, h.count + 16, (short)0);
    for (int i = 0; i < n; ++i) ++h.count[len[i]];
    if (h.count[0] == n) { h.symbol.assign(n, 0); return true; }
    short offs[16];
    offs[1] = 0;
    for (int l = 1; l < 15; ++l) offs[l + 1] = (short)(offs[l] + h.count[l]);
    h.symbol.assign(n, 0);
    for (int i = 0; i < n; ++i)
        if (len[i]) h.symbol[offs[len[i]]++] = (short)i;
    return true;
}

int decode(Bits& s, const Huff& h) {
    int code = 0, first = 0, index = 0;
    for (int l = 1; l < 16; ++l) {
        code |= (int)s.get(1);
        if (s.bad) return -1;
        int count = h.count[l];
        if (code - count < first) return h.symbol[index + (code - first)];
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    return -1;
}

const short kLenBase[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
const short kLenExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
const short kDistBase[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
const short kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

bool codes(Bits& s, std::vector<uint8_t>& out, const Huff& lit, const Huff& dist) {
    for (;;) {
        int sym = decode(s, lit);
        if (sym < 0) return false;
        if (sym < 256) { out.push_back((uint8_t)sym); continue; }
        if (sym == 256) return true;
        sym -= 257;
        if (sym >= 29) return false;
        int len = kLenBase[sym] + (int)s.get(kLenExtra[sym]);
        int ds = decode(s, dist);
        if (ds < 0 || ds >= 30) return false;
        size_t d = (size_t)kDistBase[ds] + s.get(kDistExtra[ds]);
        if (s.bad || d > out.size()) return false;
        size_t from = out.size() - d;
        for (int i = 0; i < len; ++i) out.push_back(out[from + i]);
    }
}

}  // namespace

bool zlib_inflate(const uint8_t* data, size_t n, std::vector<uint8_t>& out) {
    if (n < 2 || (data[0] & 0x0f) != 8 || ((data[0] << 8) | data[1]) % 31 != 0) return false;
    Bits s{data + 2, n - 2};
    out.clear();
    int last;
    do {
        last = (int)s.get(1);
        int type = (int)s.get(2);
        if (s.bad) return false;
        if (type == 0) {
            s.buf = 0;
            s.cnt = 0;
            if (s.pos + 4 > s.n) return false;
            unsigned len = s.p[s.pos] | (s.p[s.pos + 1] << 8);
            s.pos += 4;
            if (s.pos + len > s.n) return false;
            out.insert(out.end(), s.p + s.pos, s.p + s.pos + len);
            s.pos += len;
        } else if (type == 1) {
            static Huff lit, dist;
            static bool init = [] {
                short l[288];
                for (int i = 0; i < 144; ++i) l[i] = 8;
                for (int i = 144; i < 256; ++i) l[i] = 9;
                for (int i = 256; i < 280; ++i) l[i] = 7;
                for (int i = 280; i < 288; ++i) l[i] = 8;
                build(lit, l, 288);
                short d[30];
                for (auto& x : d) x = 5;
                build(dist, d, 30);
                return true;
            }();
            (void)init;
            if (!codes(s, out, lit, dist)) return false;
        } else if (type == 2) {
            int nlen = (int)s.get(5) + 257, ndist = (int)s.get(5) + 1, ncode = (int)s.get(4) + 4;
            static const short order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
            short lengths[320] = {0};
            for (int i = 0; i < ncode; ++i) lengths[order[i]] = (short)s.get(3);
            Huff lencode, lit, dist;
            build(lencode, lengths, 19);
            int idx = 0;
            while (idx < nlen + ndist) {
                int sym = decode(s, lencode);
                if (sym < 0) return false;
                if (sym < 16) { lengths[idx++] = (short)sym; continue; }
                short v = 0;
                int rep;
                if (sym == 16) {
                    if (idx == 0) return false;
                    v = lengths[idx - 1];
                    rep = 3 + (int)s.get(2);
                } else if (sym == 17) {
                    rep = 3 + (int)s.get(3);
                } else {
                    rep = 11 + (int)s.get(7);
                }
                if (idx + rep > nlen + ndist) return false;
                while (rep--) lengths[idx++] = v;
            }
            build(lit, lengths, nlen);
            build(dist, lengths + nlen, ndist);
            if (!codes(s, out, lit, dist)) return false;
        } else {
            return false;
        }
        if (s.bad) return false;
    } while (!last);
    return true;
}

// -------------------------------------------------------------------- PNG

static uint32_t be32(const uint8_t* p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }

static bool load_png(const std::vector<uint8_t>& f, Image& img, std::string* err) {
    auto fail = [&](const char* m) { if (err) *err = m; return false; };
    static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (f.size() < 8 || std::memcmp(f.data(), sig, 8) != 0) return fail("not a PNG file");
    uint32_t w = 0, h = 0;
    int depth = 0, ctype = 0, interlace = 0;
    std::vector<uint8_t> idat, plte, trns;
    size_t pos = 8;
    while (pos + 12 <= f.size()) {
        uint32_t len = be32(&f[pos]);
        if (pos + 12 + (size_t)len > f.size()) return fail("truncated PNG file");
        const uint8_t* type = &f[pos + 4];
        const uint8_t* d = &f[pos + 8];
        if (!std::memcmp(type, "IHDR", 4) && len >= 13) {
            w = be32(d); h = be32(d + 4); depth = d[8]; ctype = d[9]; interlace = d[12];
        } else if (!std::memcmp(type, "PLTE", 4)) {
            plte.assign(d, d + len);
        } else if (!std::memcmp(type, "tRNS", 4)) {
            trns.assign(d, d + len);
        } else if (!std::memcmp(type, "IDAT", 4)) {
            idat.insert(idat.end(), d, d + len);
        } else if (!std::memcmp(type, "IEND", 4)) {
            break;
        }
        pos += 12 + len;
    }
    if (!w || !h || w > 16384 || h > 16384) return fail("bad PNG size");
    int ch = ctype == 0 ? 1 : ctype == 2 ? 3 : ctype == 3 ? 1 : ctype == 4 ? 2 : ctype == 6 ? 4 : 0;
    if (!ch || (depth != 1 && depth != 2 && depth != 4 && depth != 8 && depth != 16)) return fail("unsupported PNG format");
    if (ctype == 3 && plte.empty()) return fail("PNG palette missing");
    std::vector<uint8_t> raw;
    if (!zlib_inflate(idat.data(), idat.size(), raw)) return fail("corrupt PNG data");

    int bits = ch * depth, bpp = std::max(1, bits / 8);
    img.w = (int)w; img.h = (int)h;
    img.rgb.assign((size_t)w * h * 3, 255);
    size_t rp = 0;
    // Adam7 passes, or one pass covering the whole image.
    static const int px0[7] = {0, 4, 0, 2, 0, 1, 0}, py0[7] = {0, 0, 4, 0, 2, 0, 1};
    static const int dx[7] = {8, 8, 4, 4, 2, 2, 1}, dy[7] = {8, 8, 8, 4, 4, 2, 2};
    int passes = interlace ? 7 : 1;
    for (int pass = 0; pass < passes; ++pass) {
        int sx = interlace ? px0[pass] : 0, sy = interlace ? py0[pass] : 0;
        int stx = interlace ? dx[pass] : 1, sty = interlace ? dy[pass] : 1;
        int pw = ((int)w - sx + stx - 1) / stx, ph = ((int)h - sy + sty - 1) / sty;
        if (pw <= 0 || ph <= 0) continue;
        size_t stride = ((size_t)pw * bits + 7) / 8;
        std::vector<uint8_t> prev(stride, 0), cur(stride);
        for (int y = 0; y < ph; ++y) {
            if (rp + 1 + stride > raw.size()) return fail("truncated PNG data");
            int filter = raw[rp++];
            for (size_t i = 0; i < stride; ++i) {
                int a = i >= (size_t)bpp ? cur[i - bpp] : 0, b = prev[i], c = i >= (size_t)bpp ? prev[i - bpp] : 0;
                int x = raw[rp + i], v;
                switch (filter) {
                    case 0: v = x; break;
                    case 1: v = x + a; break;
                    case 2: v = x + b; break;
                    case 3: v = x + ((a + b) >> 1); break;
                    case 4: {
                        int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
                        v = x + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
                        break;
                    }
                    default: return fail("bad PNG filter");
                }
                cur[i] = (uint8_t)v;
            }
            rp += stride;
            for (int x = 0; x < pw; ++x) {
                int sample[4] = {0, 0, 0, 255};
                for (int k = 0; k < ch; ++k) {
                    if (depth == 8) sample[k] = cur[(size_t)x * ch + k];
                    else if (depth == 16) sample[k] = cur[((size_t)x * ch + k) * 2];
                    else {
                        size_t bit = (size_t)x * depth;
                        int v = (cur[bit / 8] >> (8 - depth - (int)(bit % 8))) & ((1 << depth) - 1);
                        sample[k] = ctype == 3 ? v : v * 255 / ((1 << depth) - 1);
                    }
                }
                int r, g, bl, al = 255;
                if (ctype == 3) {
                    int i = sample[0];
                    if ((size_t)i * 3 + 2 >= plte.size()) i = 0;
                    r = plte[i * 3]; g = plte[i * 3 + 1]; bl = plte[i * 3 + 2];
                    if ((size_t)sample[0] < trns.size()) al = trns[sample[0]];
                } else if (ch <= 2) {
                    r = g = bl = sample[0];
                    if (ch == 2) al = sample[1];
                } else {
                    r = sample[0]; g = sample[1]; bl = sample[2];
                    if (ch == 4) al = sample[3];
                }
                size_t o = ((size_t)(sy + y * sty) * w + (size_t)(sx + x * stx)) * 3;
                img.rgb[o] = (uint8_t)((r * al + 255 * (255 - al)) / 255);
                img.rgb[o + 1] = (uint8_t)((g * al + 255 * (255 - al)) / 255);
                img.rgb[o + 2] = (uint8_t)((bl * al + 255 * (255 - al)) / 255);
            }
            std::swap(prev, cur);
        }
    }
    return true;
}

static void put32(std::vector<uint8_t>& o, uint32_t v) {
    for (int s = 24; s >= 0; s -= 8) o.push_back((uint8_t)(v >> s));
}

static void chunk(std::vector<uint8_t>& o, const char* type, const std::vector<uint8_t>& data) {
    put32(o, (uint32_t)data.size());
    size_t start = o.size();
    o.insert(o.end(), type, type + 4);
    o.insert(o.end(), data.begin(), data.end());
    put32(o, crc32(&o[start], o.size() - start));
}

bool save_png(const std::string& path, const Image& img, std::string* err) {
    if (img.w <= 0 || img.h <= 0 || img.rgb.size() != (size_t)img.w * img.h * 3) { if (err) *err = "empty image"; return false; }
    std::vector<uint8_t> raw;
    raw.reserve((size_t)img.h * (img.w * 3 + 1));
    for (int y = 0; y < img.h; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), img.rgb.begin() + (size_t)y * img.w * 3, img.rgb.begin() + (size_t)(y + 1) * img.w * 3);
    }
    // zlib stream made of stored (uncompressed) blocks
    std::vector<uint8_t> z = {0x78, 0x01};
    for (size_t p = 0; p < raw.size() || p == 0;) {
        size_t n = std::min<size_t>(65535, raw.size() - p);
        bool last = p + n >= raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back((uint8_t)(n & 0xff)); z.push_back((uint8_t)(n >> 8));
        z.push_back((uint8_t)(~n & 0xff)); z.push_back((uint8_t)((~n >> 8) & 0xff));
        z.insert(z.end(), raw.begin() + p, raw.begin() + p + n);
        p += n;
        if (last) break;
    }
    uint32_t a = 1, b = 0;
    for (uint8_t c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
    put32(z, b << 16 | a);

    std::vector<uint8_t> o = {137, 80, 78, 71, 13, 10, 26, 10};
    std::vector<uint8_t> ihdr;
    put32(ihdr, (uint32_t)img.w);
    put32(ihdr, (uint32_t)img.h);
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});
    chunk(o, "IHDR", ihdr);
    chunk(o, "IDAT", z);
    chunk(o, "IEND", {});
    std::ofstream out(upath(path), std::ios::binary);
    if (!out || !out.write((const char*)o.data(), (std::streamsize)o.size())) { if (err) *err = "cannot write file"; return false; }
    return true;
}

// -------------------------------------------------------------- BMP / PPM

static uint32_t le32(const uint8_t* p) { return p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t le16(const uint8_t* p) { return (uint16_t)(p[0] | p[1] << 8); }

static bool load_bmp(const std::vector<uint8_t>& f, Image& img, std::string* err) {
    auto fail = [&](const char* m) { if (err) *err = m; return false; };
    if (f.size() < 54 || f[0] != 'B' || f[1] != 'M') return fail("not a BMP file");
    uint32_t off = le32(&f[10]), hsize = le32(&f[14]);
    int w = (int)le32(&f[18]), h = (int)le32(&f[22]);
    int bpp = le16(&f[28]);
    uint32_t comp = le32(&f[30]);
    bool top_down = h < 0;
    h = std::abs(h);
    if (w <= 0 || h <= 0 || w > 16384 || h > 16384) return fail("bad BMP size");
    if ((comp != 0 && comp != 3) || (bpp != 24 && bpp != 32 && bpp != 8)) return fail("unsupported BMP format");
    std::vector<uint8_t> pal;
    if (bpp == 8) {
        size_t p0 = 14 + hsize;
        uint32_t n = le32(&f[46]);
        if (!n) n = 256;
        for (uint32_t i = 0; i < n && p0 + i * 4 + 3 < f.size(); ++i)
            pal.insert(pal.end(), {f[p0 + i * 4 + 2], f[p0 + i * 4 + 1], f[p0 + i * 4]});
    }
    size_t stride = ((size_t)w * bpp / 8 + 3) & ~(size_t)3;
    if (off + stride * h > f.size()) return fail("truncated BMP file");
    img.w = w; img.h = h;
    img.rgb.resize((size_t)w * h * 3);
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = &f[off + stride * (top_down ? y : h - 1 - y)];
        for (int x = 0; x < w; ++x) {
            uint8_t* o = &img.rgb[((size_t)y * w + x) * 3];
            if (bpp == 8) {
                size_t i = (size_t)row[x] * 3;
                o[0] = i + 2 < pal.size() ? pal[i] : 0; o[1] = i + 2 < pal.size() ? pal[i + 1] : 0; o[2] = i + 2 < pal.size() ? pal[i + 2] : 0;
            } else {
                const uint8_t* p = row + (size_t)x * (bpp / 8);
                o[0] = p[2]; o[1] = p[1]; o[2] = p[0];
            }
        }
    }
    return true;
}

static bool load_pnm(const std::vector<uint8_t>& f, Image& img, std::string* err) {
    auto fail = [&](const char* m) { if (err) *err = m; return false; };
    if (f.size() < 3 || f[0] != 'P' || (f[1] != '5' && f[1] != '6')) return fail("not a binary PPM/PGM file");
    size_t p = 2;
    int vals[3], got = 0;
    while (got < 3 && p < f.size()) {
        if (f[p] == '#') { while (p < f.size() && f[p] != '\n') ++p; continue; }
        if (std::isspace(f[p])) { ++p; continue; }
        int v = 0;
        while (p < f.size() && std::isdigit(f[p])) v = v * 10 + (f[p++] - '0');
        vals[got++] = v;
    }
    ++p;
    int w = vals[0], h = vals[1], mx = vals[2], ch = f[1] == '6' ? 3 : 1;
    if (got < 3 || w <= 0 || h <= 0 || mx <= 0 || mx > 255) return fail("unsupported PPM/PGM file");
    if (p + (size_t)w * h * ch > f.size()) return fail("truncated PPM/PGM file");
    img.w = w; img.h = h;
    img.rgb.resize((size_t)w * h * 3);
    for (size_t i = 0; i < (size_t)w * h; ++i)
        for (int k = 0; k < 3; ++k) img.rgb[i * 3 + k] = (uint8_t)(f[p + i * ch + (ch == 3 ? k : 0)] * 255 / mx);
    return true;
}

bool load_image(const std::string& path, Image& img, std::string* err) {
    std::vector<uint8_t> f;
    if (!read_all(path, f, err)) return false;
    if (f.size() >= 8 && f[0] == 137 && f[1] == 'P') return load_png(f, img, err);
    if (f.size() >= 2 && f[0] == 'B' && f[1] == 'M') return load_bmp(f, img, err);
    if (f.size() >= 2 && f[0] == 'P') return load_pnm(f, img, err);
    if (err) *err = "unsupported image format (use PNG, BMP, PPM or PGM)";
    return false;
}

Image square_thumbnail(const Image& img, int size) {
    Image out;
    out.w = out.h = size;
    out.rgb.assign((size_t)size * size * 3, 0);
    int side = std::min(img.w, img.h);
    if (side <= 0 || size <= 0) return out;
    int x0 = (img.w - side) / 2, y0 = (img.h - side) / 2;
    double cell = (double)side / size;
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x) {
            // Average of the source pixels this output pixel covers (box filter).
            int sx0 = x0 + (int)(x * cell), sx1 = std::max(sx0 + 1, x0 + (int)((x + 1) * cell));
            int sy0 = y0 + (int)(y * cell), sy1 = std::max(sy0 + 1, y0 + (int)((y + 1) * cell));
            long sum[3] = {0, 0, 0}, n = 0;
            for (int sy = sy0; sy < sy1 && sy < img.h; ++sy)
                for (int sx = sx0; sx < sx1 && sx < img.w; ++sx) {
                    const uint8_t* p = &img.rgb[((size_t)sy * img.w + sx) * 3];
                    sum[0] += p[0]; sum[1] += p[1]; sum[2] += p[2];
                    ++n;
                }
            for (int k = 0; k < 3; ++k) out.rgb[((size_t)y * size + x) * 3 + k] = (uint8_t)(n ? sum[k] / n : 0);
        }
    return out;
}

// -------------------------------------------------------------------- WAV

bool load_wav(const std::string& path, Sound& snd, std::string* err) {
    auto fail = [&](const char* m) { if (err) *err = m; return false; };
    std::vector<uint8_t> f;
    if (!read_all(path, f, err)) return false;
    if (f.size() < 12 || std::memcmp(&f[0], "RIFF", 4) || std::memcmp(&f[8], "WAVE", 4)) return fail("not a WAV file");
    int fmt = 0, channels = 0, rate = 0, bits = 0;
    const uint8_t* data = nullptr;
    size_t data_len = 0;
    for (size_t p = 12; p + 8 <= f.size();) {
        uint32_t len = le32(&f[p + 4]);
        const uint8_t* d = &f[p + 8];
        size_t avail = std::min<size_t>(len, f.size() - p - 8);
        if (!std::memcmp(&f[p], "fmt ", 4) && avail >= 16) {
            fmt = le16(d); channels = le16(d + 2); rate = (int)le32(d + 4); bits = le16(d + 14);
            if (fmt == 0xFFFE && avail >= 26) fmt = le16(d + 24);
        } else if (!std::memcmp(&f[p], "data", 4)) {
            data = d;
            data_len = avail;
        }
        p += 8 + len + (len & 1);
    }
    if (!data || channels <= 0 || rate <= 0) return fail("WAV file has no audio");
    bool ok = (fmt == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) || (fmt == 3 && bits == 32);
    if (!ok) return fail("unsupported WAV format (use PCM or 32-bit float)");
    int bps = bits / 8;
    size_t frames = data_len / ((size_t)bps * channels);
    snd.rate = rate;
    snd.samples.resize(frames);
    for (size_t i = 0; i < frames; ++i) {
        float sum = 0;
        for (int c = 0; c < channels; ++c) {
            const uint8_t* s = data + (i * channels + c) * bps;
            float v;
            if (fmt == 3) { std::memcpy(&v, s, 4); }
            else if (bits == 8) v = (s[0] - 128) / 128.f;
            else if (bits == 16) v = (int16_t)le16(s) / 32768.f;
            else if (bits == 24) v = (float)((int32_t)((uint32_t)s[0] << 8 | (uint32_t)s[1] << 16 | (uint32_t)s[2] << 24) >> 8) / 8388608.f;
            else v = (float)(int32_t)le32(s) / 2147483648.f;
            sum += v;
        }
        snd.samples[i] = std::max(-1.f, std::min(1.f, sum / channels));
    }
    return true;
}

bool save_wav(const std::string& path, const Sound& snd, std::string* err) {
    std::vector<uint8_t> o;
    auto u32 = [&](uint32_t v) { for (int s = 0; s < 32; s += 8) o.push_back((uint8_t)(v >> s)); };
    auto u16 = [&](uint16_t v) { o.push_back((uint8_t)v); o.push_back((uint8_t)(v >> 8)); };
    uint32_t bytes = (uint32_t)snd.samples.size() * 2;
    o.insert(o.end(), {'R', 'I', 'F', 'F'}); u32(36 + bytes);
    o.insert(o.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '}); u32(16);
    u16(1); u16(1); u32((uint32_t)snd.rate); u32((uint32_t)snd.rate * 2); u16(2); u16(16);
    o.insert(o.end(), {'d', 'a', 't', 'a'}); u32(bytes);
    for (float v : snd.samples) u16((uint16_t)(int16_t)std::lround(std::max(-1.f, std::min(1.f, v)) * 32767.f));
    std::ofstream out(upath(path), std::ios::binary);
    if (!out || !out.write((const char*)o.data(), (std::streamsize)o.size())) { if (err) *err = "cannot write file"; return false; }
    return true;
}

std::vector<float> resample(const std::vector<float>& x, int from, int to) {
    if (from == to || x.empty() || from <= 0 || to <= 0) return x;
    size_t n = (size_t)((double)x.size() * to / from);
    std::vector<float> y(n);
    double step = (double)from / to;
    if (to < from) {
        // Average the source samples of each output sample (a simple low-pass).
        for (size_t i = 0; i < n; ++i) {
            size_t a = (size_t)(i * step), b = std::min(x.size(), std::max(a + 1, (size_t)((i + 1) * step)));
            double s = 0;
            for (size_t k = a; k < b; ++k) s += x[k];
            y[i] = (float)(s / (double)(b - a));
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            double t = i * step;
            size_t a = (size_t)t;
            double f = t - a;
            float v0 = x[std::min(a, x.size() - 1)], v1 = x[std::min(a + 1, x.size() - 1)];
            y[i] = (float)(v0 + (v1 - v0) * f);
        }
    }
    return y;
}

int mulaw_encode(float x) {
    const float mu = 255.f;
    x = std::max(-1.f, std::min(1.f, x));
    float y = (x < 0 ? -1.f : 1.f) * std::log1p(mu * std::fabs(x)) / std::log1p(mu);
    return std::max(0, std::min(255, (int)std::lround((y + 1.f) * 127.5f)));
}

float mulaw_decode(int level) {
    const float mu = 255.f;
    float y = level / 127.5f - 1.f;
    return (y < 0 ? -1.f : 1.f) * (std::pow(1.f + mu, std::fabs(y)) - 1.f) / mu;
}

// ------------------------------------------------------------- demo data

static bool make_dir(const std::string& dir, std::string* err) {
    std::error_code ec;
    std::filesystem::create_directories(upath(dir), ec);
    if (ec || !std::filesystem::is_directory(upath(dir), ec)) { if (err) *err = "cannot create folder"; return false; }
    return true;
}

bool write_demo_images(const std::string& dir, int count, std::string* err) {
    if (!make_dir(dir, err)) return false;
    // A few backgrounds (sky, sunset, grass, night) and bright shapes.
    static const uint8_t bg[4][2][3] = {{{90, 160, 230}, {200, 230, 250}}, {{250, 150, 70}, {120, 60, 140}},
                                        {{150, 210, 120}, {40, 120, 50}}, {{20, 25, 60}, {70, 60, 120}}};
    static const uint8_t fg[6][3] = {{230, 40, 40}, {250, 210, 40}, {40, 90, 220}, {250, 250, 250}, {30, 30, 30}, {170, 60, 200}};
    const int S = 32;
    Rng rng(2026);
    for (int k = 0; k < count; ++k) {
        Image img;
        img.w = img.h = S;
        img.rgb.resize(S * S * 3);
        const auto& b = bg[rng.below(4)];
        for (int y = 0; y < S; ++y)
            for (int x = 0; x < S; ++x)
                for (int c = 0; c < 3; ++c) img.rgb[(y * S + x) * 3 + c] = (uint8_t)(b[0][c] + (b[1][c] - b[0][c]) * y / (S - 1));
        int shapes = 1 + rng.below(2);
        for (int n = 0; n < shapes; ++n) {
            int kind = rng.below(3);
            const uint8_t* col = fg[rng.below(6)];
            float cx = 6 + rng.uniform() * (S - 12), cy = 6 + rng.uniform() * (S - 12), r = 4 + rng.uniform() * 7;
            for (int y = 0; y < S; ++y)
                for (int x = 0; x < S; ++x) {
                    float dx = x + 0.5f - cx, dy = y + 0.5f - cy;
                    bool in = kind == 0 ? dx * dx + dy * dy <= r * r
                            : kind == 1 ? std::fabs(dx) <= r * 0.8f && std::fabs(dy) <= r * 0.8f
                                        : std::fabs(dx) + std::fabs(dy) <= r;
                    if (in) std::memcpy(&img.rgb[(y * S + x) * 3], col, 3);
                }
        }
        char name[32];
        std::snprintf(name, sizeof name, "picture-%03d.png", k + 1);
        if (!save_png((upath(dir) / name).u8string(), img, err)) return false;
    }
    return true;
}

bool write_demo_sounds(const std::string& dir, int count, std::string* err) {
    if (!make_dir(dir, err)) return false;
    // Short melodies on a pentatonic scale, a plucked tone with an overtone.
    static const float scale[] = {261.6f, 293.7f, 329.6f, 392.0f, 440.0f, 523.3f, 587.3f, 659.3f};
    const int rate = 16000;
    Rng rng(7);
    for (int k = 0; k < count; ++k) {
        Sound snd;
        snd.rate = rate;
        int notes = 6 + rng.below(4);
        float beat = 0.2f + 0.05f * rng.below(3);
        int step = 0;
        for (int n = 0; n < notes; ++n) {
            step = std::max(0, std::min(7, step + (int)rng.below(5) - 2));
            float f = scale[step];
            int len = (int)(beat * rate);
            for (int i = 0; i < len; ++i) {
                float t = (float)i / rate;
                float env = std::exp(-t * 6.f) * std::min(1.f, t * 200.f);
                snd.samples.push_back(0.5f * env * (std::sin(6.2831853f * f * t) + 0.3f * std::sin(6.2831853f * 2 * f * t)));
            }
        }
        char name[32];
        std::snprintf(name, sizeof name, "melody-%03d.wav", k + 1);
        if (!save_wav((upath(dir) / name).u8string(), snd, err)) return false;
    }
    return true;
}

}  // namespace kor
