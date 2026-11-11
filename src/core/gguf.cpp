// Kor Studio - GGUF language models: file reader/writer, quantized weights,
// tokenizers, chat formats, inference and LoRA training.
#include "gguf.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <queue>
#include <unordered_map>
#include "kernels.h"
#include "parallel.h"
#include "sysinfo.h"
#include "trainer.h"
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace kor {

namespace gbase { void dequant_blocks(int type, const uint8_t* x, int64_t nb, float* y); }
#ifdef KOR_HAVE_AVX2_TU
namespace gavx2 { void dequant_blocks(int type, const uint8_t* x, int64_t nb, float* y); }
#endif

namespace {

// ------------------------------------------------------------ number formats

uint16_t f2h(float v) {
    uint32_t x;
    std::memcpy(&x, &v, 4);
    uint32_t s = (x >> 16) & 0x8000, m = x & 0x7FFFFF;
    int ex = (int)((x >> 23) & 0xFF);
    if (ex == 0xFF) return (uint16_t)(s | 0x7C00 | (m ? 0x200 : 0));
    int e = ex - 127 + 15;
    if (e >= 31) return (uint16_t)(s | 0x7C00);
    if (e <= 0) {
        if (e < -10) return (uint16_t)s;
        m |= 0x800000;
        int shift = 14 - e;
        uint32_t h = m >> shift, rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) ++h;
        return (uint16_t)(s | h);
    }
    uint32_t h = ((uint32_t)e << 10) | (m >> 13), rem = m & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) ++h;
    return (uint16_t)(s | h);
}


struct TypeInfo { int type; const char* name; int bs; size_t bytes; };
const TypeInfo kTypes[] = {
    {GGML_F32, "F32", 1, 4},       {GGML_F16, "F16", 1, 2},       {GGML_BF16, "BF16", 1, 2},
    {GGML_Q4_0, "Q4_0", 32, 18},   {GGML_Q4_1, "Q4_1", 32, 20},   {GGML_Q5_0, "Q5_0", 32, 22},
    {GGML_Q5_1, "Q5_1", 32, 24},   {GGML_Q8_0, "Q8_0", 32, 34},   {GGML_Q2_K, "Q2_K", 256, 84},
    {GGML_Q3_K, "Q3_K", 256, 110}, {GGML_Q4_K, "Q4_K", 256, 144}, {GGML_Q5_K, "Q5_K", 256, 176},
    {GGML_Q6_K, "Q6_K", 256, 210},
};
const TypeInfo* type_info(int t) {
    for (const TypeInfo& i : kTypes)
        if (i.type == t) return &i;
    return nullptr;
}

using DequantFn = void (*)(int, const uint8_t*, int64_t, float*);
DequantFn dequant_fn() {
#ifdef KOR_HAVE_AVX2_TU
    static const DequantFn f = cpu_has_avx2() ? &gavx2::dequant_blocks : &gbase::dequant_blocks;
#else
    static const DequantFn f = &gbase::dequant_blocks;
#endif
    return f;
}

void dequant_blocks(int type, const uint8_t* x, int64_t nb, float* y) { dequant_fn()(type, x, nb, y); }

// ------------------------------------------------------------ mapped files

class MappedFile {
public:
    const uint8_t* data = nullptr;
    size_t size = 0;
    MappedFile() = default;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    ~MappedFile() { close(); }

    bool open(const std::string& path, std::string* err) {
        close();
#if defined(_WIN32)
        std::wstring w = std::filesystem::u8path(path).wstring();
        file_ = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) {
            if (err) *err = "cannot open " + path;
            return false;
        }
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(file_, &sz) || sz.QuadPart == 0) {
            if (err) *err = "empty file: " + path;
            close();
            return false;
        }
        size = (size_t)sz.QuadPart;
        map_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (map_) data = (const uint8_t*)MapViewOfFile(map_, FILE_MAP_READ, 0, 0, 0);
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            if (err) *err = "cannot open " + path;
            return false;
        }
        struct stat st;
        if (fstat(fd_, &st) != 0 || st.st_size == 0) {
            if (err) *err = "empty file: " + path;
            close();
            return false;
        }
        size = (size_t)st.st_size;
        void* p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd_, 0);
        if (p != MAP_FAILED) data = (const uint8_t*)p;
#endif
        if (!data) {
            if (err) *err = "cannot map " + path;
            close();
            return false;
        }
        return true;
    }

    void close() {
#if defined(_WIN32)
        if (data) UnmapViewOfFile(data);
        if (map_) CloseHandle(map_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
        map_ = nullptr;
        file_ = INVALID_HANDLE_VALUE;
#else
        if (data) munmap((void*)data, size);
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
#endif
        data = nullptr;
        size = 0;
    }

private:
#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE, map_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// ------------------------------------------------------------ GGUF reading

enum { GV_U8, GV_I8, GV_U16, GV_I16, GV_U32, GV_I32, GV_F32, GV_BOOL, GV_STR, GV_ARR, GV_U64, GV_I64, GV_F64 };

struct GgufKv {
    int type = -1, arr = -1;
    double num = 0;
    std::string str;
    std::vector<std::string> strs;
    std::vector<double> nums;
};

struct GgufTensor {
    std::string name;
    std::vector<int64_t> dims;
    int type = 0;
    const uint8_t* data = nullptr;
    size_t bytes = 0;
    int64_t count() const {
        int64_t n = 1;
        for (int64_t d : dims) n *= d;
        return n;
    }
};

struct Cursor {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;
    template <class T> T rd() {
        T v{};
        if ((size_t)(end - p) < sizeof(T)) { ok = false; p = end; return v; }
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        uint64_t n = rd<uint64_t>();
        if (!ok || n > (uint64_t)(end - p)) { ok = false; p = end; return {}; }
        std::string s((const char*)p, (size_t)n);
        p += n;
        return s;
    }
};

double read_num(Cursor& c, int t) {
    switch (t) {
    case GV_U8: return c.rd<uint8_t>();
    case GV_I8: return c.rd<int8_t>();
    case GV_U16: return c.rd<uint16_t>();
    case GV_I16: return c.rd<int16_t>();
    case GV_U32: return c.rd<uint32_t>();
    case GV_I32: return c.rd<int32_t>();
    case GV_F32: return c.rd<float>();
    case GV_BOOL: return c.rd<uint8_t>() ? 1 : 0;
    case GV_U64: return (double)c.rd<uint64_t>();
    case GV_I64: return (double)c.rd<int64_t>();
    case GV_F64: return c.rd<double>();
    default: c.ok = false; return 0;
    }
}

bool read_value(Cursor& c, int t, GgufKv* out, int depth = 0) {
    if (t == GV_STR) {
        std::string s = c.str();
        if (out) out->str = std::move(s);
        return c.ok;
    }
    if (t == GV_ARR) {
        int at = (int)c.rd<uint32_t>();
        uint64_t n = c.rd<uint64_t>();
        if (!c.ok || n > (uint64_t)(c.end - c.p) || depth > 2) return c.ok = false;
        if (out) out->arr = at;
        if (at == GV_STR) {
            if (out) out->strs.reserve((size_t)n);
            for (uint64_t i = 0; i < n && c.ok; ++i) {
                std::string s = c.str();
                if (out) out->strs.push_back(std::move(s));
            }
        } else if (at == GV_ARR) {
            for (uint64_t i = 0; i < n && c.ok; ++i) read_value(c, GV_ARR, nullptr, depth + 1);
        } else {
            if (out) out->nums.reserve((size_t)n);
            for (uint64_t i = 0; i < n && c.ok; ++i) {
                double v = read_num(c, at);
                if (out) out->nums.push_back(v);
            }
        }
        return c.ok;
    }
    double v = read_num(c, t);
    if (out) out->num = v;
    return c.ok;
}

struct GgufFile {
    MappedFile f;
    std::map<std::string, GgufKv> kv;
    std::map<std::string, GgufTensor> tensors;

    bool open(const std::string& path, std::string* err) {
        if (!f.open(path, err)) return false;
        auto fail = [&](const std::string& m) {
            if (err) *err = m;
            return false;
        };
        Cursor c{f.data, f.data + f.size};
        if (f.size < 24 || std::memcmp(f.data, "GGUF", 4) != 0) return fail("not a GGUF file");
        c.p += 4;
        uint32_t version = c.rd<uint32_t>();
        if (version < 2 || version > 3) return fail("unsupported GGUF version " + std::to_string(version));
        uint64_t nt = c.rd<uint64_t>(), nkv = c.rd<uint64_t>();
        if (nt > 1000000 || nkv > 1000000) return fail("damaged GGUF header");
        for (uint64_t i = 0; i < nkv && c.ok; ++i) {
            std::string key = c.str();
            GgufKv v;
            v.type = (int)c.rd<uint32_t>();
            if (!c.ok || !read_value(c, v.type, &v)) return fail("damaged GGUF metadata");
            kv[key] = std::move(v);
        }
        std::vector<std::pair<GgufTensor, uint64_t>> list;
        for (uint64_t i = 0; i < nt && c.ok; ++i) {
            GgufTensor t;
            t.name = c.str();
            uint32_t nd = c.rd<uint32_t>();
            if (nd == 0 || nd > 4) return fail("damaged tensor info: " + t.name);
            for (uint32_t k = 0; k < nd; ++k) {
                uint64_t d = c.rd<uint64_t>();
                if (d == 0 || d > (1ull << 40)) return fail("damaged tensor info: " + t.name);
                t.dims.push_back((int64_t)d);
            }
            t.type = (int)c.rd<uint32_t>();
            uint64_t off = c.rd<uint64_t>();
            list.push_back({std::move(t), off});
        }
        if (!c.ok) return fail("damaged GGUF file");
        uint64_t align = (uint64_t)num("general.alignment", 32);
        if (align == 0 || (align & (align - 1))) align = 32;
        uint64_t base = (uint64_t)(c.p - f.data);
        base = (base + align - 1) / align * align;
        for (auto& [t, off] : list) {
            const TypeInfo* ti = type_info(t.type);
            if (ti && t.dims[0] % ti->bs == 0) {
                t.bytes = ggml_row_bytes(t.type, t.dims[0]) * (size_t)(t.count() / t.dims[0]);
                if (base + off + t.bytes > f.size) return fail("the file is cut short (" + t.name + ")");
                t.data = f.data + base + off;
            }
            tensors[t.name] = std::move(t);
        }
        return true;
    }

    const GgufKv* get(const std::string& key) const {
        auto it = kv.find(key);
        return it == kv.end() ? nullptr : &it->second;
    }
    double num(const std::string& key, double def) const {
        const GgufKv* v = get(key);
        return v && v->type != GV_STR && v->type != GV_ARR ? v->num : def;
    }
    std::string str(const std::string& key, const std::string& def = "") const {
        const GgufKv* v = get(key);
        return v && v->type == GV_STR ? v->str : def;
    }
    const GgufTensor* tensor(const std::string& name) const {
        auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : &it->second;
    }
};

// ------------------------------------------------------------ text helpers

uint32_t utf8_at(const std::string& s, size_t i, int* len) {
    unsigned char c = (unsigned char)s[i];
    int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (n == 0 || i + n > s.size()) { *len = 1; return 0xFFFD; }
    uint32_t cp = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
    for (int k = 1; k < n; ++k) {
        unsigned char d = (unsigned char)s[i + k];
        if ((d >> 6) != 2) { *len = 1; return 0xFFFD; }
        cp = (cp << 6) | (d & 0x3F);
    }
    *len = n;
    return cp;
}

std::string utf8_of(uint32_t cp) {
    std::string o;
    if (cp < 0x80) {
        o += (char)cp;
    } else if (cp < 0x800) {
        o += (char)(0xC0 | (cp >> 6));
        o += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        o += (char)(0xE0 | (cp >> 12));
        o += (char)(0x80 | ((cp >> 6) & 0x3F));
        o += (char)(0x80 | (cp & 0x3F));
    } else {
        o += (char)(0xF0 | (cp >> 18));
        o += (char)(0x80 | ((cp >> 12) & 0x3F));
        o += (char)(0x80 | ((cp >> 6) & 0x3F));
        o += (char)(0x80 | (cp & 0x3F));
    }
    return o;
}

// The length of the longest prefix of s made of whole UTF-8 characters.
size_t utf8_whole(const std::string& s) {
    size_t n = s.size(), i = n;
    int back = 0;
    while (i > 0 && back < 4) {
        unsigned char c = (unsigned char)s[i - 1];
        if ((c >> 6) != 2) {
            int need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
            return (int)(n - (i - 1)) >= need ? n : i - 1;
        }
        --i;
        ++back;
    }
    return n;
}

bool is_space(uint32_t c) {
    return c == ' ' || (c >= 9 && c <= 13) || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
bool is_number(uint32_t c) {
    return (c >= '0' && c <= '9') || c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0xBC && c <= 0xBE) ||
           (c >= 0x660 && c <= 0x669) || (c >= 0x6F0 && c <= 0x6F9) || (c >= 0x966 && c <= 0x96F) ||
           (c >= 0x2070 && c <= 0x2079) || (c >= 0x2080 && c <= 0x2089) || (c >= 0x2150 && c <= 0x2189) ||
           (c >= 0x2460 && c <= 0x249B) || (c >= 0x3007 && c <= 0x3007) || (c >= 0xFF10 && c <= 0xFF19);
}
// An approximation of \p{L}: everything that is not a space, number, mark,
// punctuation or symbol of the common blocks.
bool is_letter(uint32_t c) {
    if (c < 0x80) return (c | 0x20) >= 'a' && (c | 0x20) <= 'z';
    if (c < 0xC0) return c == 0xAA || c == 0xB5 || c == 0xBA;
    if (c == 0xD7 || c == 0xF7) return false;
    if (is_space(c) || is_number(c)) return false;
    if (c >= 0x300 && c <= 0x36F) return false;    // combining marks
    if (c >= 0x2000 && c <= 0x2BFF) return false;  // punctuation, symbols, arrows, shapes
    if (c >= 0x3000 && c <= 0x303F) return false;  // CJK punctuation
    if (c >= 0xFE30 && c <= 0xFE4F) return false;
    if ((c >= 0xFF00 && c <= 0xFF20) || (c >= 0xFF3B && c <= 0xFF40) || (c >= 0xFF5B && c <= 0xFF65)) return false;
    if (c >= 0x1F000 && c <= 0x1FAFF) return false;  // emoji
    if (c == 0xFFFD || (c >= 0xE000 && c <= 0xF8FF)) return false;
    return true;
}

// ------------------------------------------------------------ tokenizer

enum TokType { TT_NORMAL = 1, TT_UNKNOWN = 2, TT_CONTROL = 3, TT_USER = 4, TT_UNUSED = 5, TT_BYTE = 6 };

struct LlmTokenizer {
    bool spm = false;
    std::vector<std::string> toks;
    std::vector<float> scores;
    std::vector<int> types;
    std::unordered_map<std::string, int> index;
    std::unordered_map<std::string, int> merges;
    std::vector<std::vector<int>> special_by_byte;  // specials by first byte, longest first
    std::vector<char> end;
    int bos = -1, eos = -1, unk = -1, byte_tok[256];
    bool add_bos = false, add_space = true;
    bool llama3_split = false;  // the llama-3 / qwen2 pre-tokenizer instead of gpt-2's
    int digits = 0;             // longest digit group (0 = any)
    uint32_t byte_cp[256];
    std::unordered_map<uint32_t, int> cp_byte;

    void init_bytes() {
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            bool keep = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            byte_cp[b] = keep ? (uint32_t)b : (uint32_t)(256 + n++);
            cp_byte[byte_cp[b]] = b;
        }
    }

    int find(const std::string& s) const {
        auto it = index.find(s);
        return it == index.end() ? -1 : it->second;
    }

    // Splits text into pieces with the pre-tokenizer pattern of the model.
    std::vector<std::string> pretokenize(const std::string& text) const {
        std::vector<uint32_t> cp;
        std::vector<size_t> at;
        for (size_t i = 0; i < text.size();) {
            int len;
            cp.push_back(utf8_at(text, i, &len));
            at.push_back(i);
            i += len;
        }
        size_t n = cp.size();
        at.push_back(text.size());
        auto L = [&](size_t i) { return i < n && is_letter(cp[i]); };
        auto N = [&](size_t i) { return i < n && is_number(cp[i]); };
        auto S = [&](size_t i) { return i < n && is_space(cp[i]); };
        auto nl = [&](size_t i) { return i < n && (cp[i] == '\r' || cp[i] == '\n'); };
        auto lower = [&](size_t i) { return i < n && cp[i] < 128 ? (uint32_t)std::tolower((int)cp[i]) : (i < n ? cp[i] : 0); };
        std::vector<std::string> out;
        size_t i = 0;
        while (i < n) {
            size_t e = i;
            // contractions: 's 't 'm 'd 're 've 'll
            if (cp[i] == '\'' && i + 1 < n) {
                uint32_t a = llama3_split ? lower(i + 1) : cp[i + 1];
                uint32_t b = llama3_split ? lower(i + 2) : (i + 2 < n ? cp[i + 2] : 0);
                if (a == 's' || a == 't' || a == 'm' || a == 'd') e = i + 2;
                else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) e = i + 3;
            }
            if (e == i) {
                if (llama3_split) {
                    if (L(i) || (!nl(i) && !L(i) && !N(i) && L(i + 1))) {
                        e = i + 1;
                        while (L(e)) ++e;
                    } else if (N(i)) {
                        e = i;
                        while (N(e) && (digits == 0 || (int)(e - i) < digits)) ++e;
                    }
                } else {
                    size_t j = cp[i] == ' ' ? i + 1 : i;
                    if (L(j)) {
                        e = j;
                        while (L(e)) ++e;
                    } else if (N(digits ? i : j)) {
                        size_t k = digits ? i : j;
                        e = k;
                        while (N(e) && (digits == 0 || (int)(e - k) < digits)) ++e;
                    }
                }
            }
            if (e == i) {  // " ?[^\s\p{L}\p{N}]+" (llama-3: followed by newlines)
                size_t j = cp[i] == ' ' ? i + 1 : i;
                if (j < n && !S(j) && !L(j) && !N(j)) {
                    e = j;
                    while (e < n && !S(e) && !L(e) && !N(e)) ++e;
                    if (llama3_split)
                        while (nl(e)) ++e;
                }
            }
            if (e == i && S(i)) {
                size_t r = i;
                while (S(r)) ++r;
                if (llama3_split) {  // \s*[\r\n]+
                    size_t last = i;
                    for (size_t k = i; k < r; ++k)
                        if (nl(k)) last = k + 1;
                    if (last > i) e = last;
                }
                if (e == i) e = (r == n || r - i == 1) ? r : r - 1;  // \s+(?!\S) then \s+
            }
            if (e == i) e = i + 1;
            out.push_back(text.substr(at[i], at[e] - at[i]));
            i = e;
        }
        return out;
    }

    void bpe_word(const std::string& word, std::vector<int>& ids) const {
        std::vector<std::string> sym;
        for (unsigned char b : word) sym.push_back(utf8_of(byte_cp[b]));
        if (sym.empty()) return;
        std::string whole;
        for (auto& s : sym) whole += s;
        int direct = find(whole);
        if (direct >= 0) { ids.push_back(direct); return; }
        std::string key;
        while (sym.size() > 1) {
            int best = INT32_MAX;
            size_t bi = 0;
            for (size_t i = 0; i + 1 < sym.size(); ++i) {
                key.assign(sym[i]).append(" ").append(sym[i + 1]);
                auto it = merges.find(key);
                if (it != merges.end() && it->second < best) { best = it->second; bi = i; }
            }
            if (best == INT32_MAX) break;
            sym[bi] += sym[bi + 1];
            sym.erase(sym.begin() + bi + 1);
        }
        for (const std::string& s : sym) {
            int id = find(s);
            if (id >= 0) { ids.push_back(id); continue; }
            for (size_t i = 0; i < s.size();) {
                int len;
                utf8_at(s, i, &len);
                int c = find(s.substr(i, len));
                if (c >= 0) ids.push_back(c);
                else if (unk >= 0) ids.push_back(unk);
                i += len;
            }
        }
    }

    void spm_text(const std::string& raw, std::vector<int>& ids) const {
        std::string s;
        for (char c : raw) {
            if (c == ' ') s += "\xE2\x96\x81";
            else s += c;
        }
        struct Sym { int prev, next; size_t start, len; };
        std::vector<Sym> sy;
        for (size_t i = 0; i < s.size();) {
            int len;
            utf8_at(s, i, &len);
            sy.push_back({(int)sy.size() - 1, (int)sy.size() + 1, i, (size_t)len});
            i += len;
        }
        if (sy.empty()) return;
        sy.back().next = -1;
        struct Big { int left, right; float score; size_t len; };
        auto cmp = [](const Big& a, const Big& b) { return a.score < b.score || (a.score == b.score && a.left > b.left); };
        std::priority_queue<Big, std::vector<Big>, decltype(cmp)> q(cmp);
        auto add = [&](int l, int r) {
            if (l < 0 || r < 0) return;
            int id = find(s.substr(sy[l].start, sy[l].len + sy[r].len));
            if (id < 0) return;
            q.push({l, r, scores.empty() ? 0.f : scores[id], sy[l].len + sy[r].len});
        };
        for (size_t i = 1; i < sy.size(); ++i) add((int)i - 1, (int)i);
        while (!q.empty()) {
            Big b = q.top();
            q.pop();
            Sym& l = sy[b.left];
            Sym& r = sy[b.right];
            if (l.len == 0 || r.len == 0 || l.len + r.len != b.len || l.next != b.right) continue;
            l.len += r.len;
            r.len = 0;
            l.next = r.next;
            if (r.next >= 0) sy[r.next].prev = b.left;
            add(l.prev, b.left);
            add(b.left, l.next);
        }
        for (int i = 0; i >= 0; i = sy[i].next) {
            std::string piece = s.substr(sy[i].start, sy[i].len);
            int id = find(piece);
            if (id >= 0) { ids.push_back(id); continue; }
            for (unsigned char c : piece) {
                if (byte_tok[c] >= 0) ids.push_back(byte_tok[c]);
                else if (unk >= 0) ids.push_back(unk);
            }
        }
    }

    std::vector<int> encode(const std::string& text, bool bos_first, bool special) const {
        std::vector<int> ids;
        if (bos_first && bos >= 0) ids.push_back(bos);
        bool prev_special = true;
        auto raw = [&](const std::string& t) {
            if (t.empty()) return;
            if (spm) {
                spm_text(add_space && prev_special ? " " + t : t, ids);
            } else {
                for (const std::string& w : pretokenize(t)) bpe_word(w, ids);
            }
            prev_special = false;
        };
        size_t start = 0, i = 0;
        while (special && i < text.size()) {
            int hit = -1;
            for (int id : special_by_byte[(unsigned char)text[i]])
                if (text.compare(i, toks[id].size(), toks[id]) == 0) { hit = id; break; }
            if (hit < 0) { ++i; continue; }
            raw(text.substr(start, i - start));
            ids.push_back(hit);
            prev_special = true;
            i += toks[hit].size();
            start = i;
        }
        raw(text.substr(start));
        return ids;
    }

    std::string piece(int id) const {
        if (id < 0 || id >= (int)toks.size()) return "";
        int t = types[id];
        const std::string& s = toks[id];
        if (t == TT_CONTROL || t == TT_UNUSED || t == TT_UNKNOWN) return "";
        if (t == TT_BYTE) {
            if (s.size() == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x') return std::string(1, (char)std::strtol(s.c_str() + 3, nullptr, 16));
            return "";
        }
        if (spm || t == TT_USER) {
            if (!spm) return s;
            std::string o;
            for (size_t i = 0; i < s.size(); ++i) {
                if (s.compare(i, 3, "\xE2\x96\x81") == 0) { o += ' '; i += 2; }
                else o += s[i];
            }
            return o;
        }
        std::string o;
        for (size_t i = 0; i < s.size();) {
            int len;
            uint32_t cp = utf8_at(s, i, &len);
            auto it = cp_byte.find(cp);
            if (it != cp_byte.end()) o += (char)it->second;
            else o += s.substr(i, len);
            i += len;
        }
        return o;
    }
};

enum ChatKind { CK_PLAIN, CK_CHATML, CK_LLAMA3, CK_MISTRAL, CK_PHI3, CK_ZEPHYR, CK_GEMMA };
const char* const kChatNames[] = {"plain", "ChatML", "Llama 3", "Mistral [INST]", "Phi-3", "Zephyr", "Gemma"};

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

const char* const kLoraNames[7] = {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"};

}  // namespace

// ------------------------------------------------------------ public helpers

const char* ggml_type_name(int type) {
    const TypeInfo* t = type_info(type);
    return t ? t->name : nullptr;
}
int ggml_block_size(int type) {
    const TypeInfo* t = type_info(type);
    return t ? t->bs : 0;
}
size_t ggml_block_bytes(int type) {
    const TypeInfo* t = type_info(type);
    return t ? t->bytes : 0;
}
size_t ggml_row_bytes(int type, int64_t n) {
    const TypeInfo* t = type_info(type);
    return t ? (size_t)(n / t->bs) * t->bytes : 0;
}

void ggml_dequantize(int type, const uint8_t* src, int64_t n, float* out) {
    int bs = ggml_block_size(type);
    if (bs > 0) dequant_blocks(type, src, n / bs, out);
}

bool ggml_quantize(int type, const float* x, int64_t n, std::vector<uint8_t>& out) {
    int bs = ggml_block_size(type);
    if (bs == 0 || n % bs) return false;
    out.assign(ggml_row_bytes(type, n), 0);
    uint8_t* o = out.data();
    switch (type) {
    case GGML_F32:
        std::memcpy(o, x, (size_t)n * 4);
        return true;
    case GGML_F16:
        for (int64_t i = 0; i < n; ++i) { uint16_t h = f2h(x[i]); std::memcpy(o + 2 * i, &h, 2); }
        return true;
    case GGML_Q8_0:
        for (int64_t b = 0; b < n / 32; ++b, x += 32, o += 34) {
            float amax = 0;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
            float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
            uint16_t h = f2h(d);
            std::memcpy(o, &h, 2);
            for (int j = 0; j < 32; ++j) o[2 + j] = (uint8_t)(int8_t)std::lround(x[j] * id);
        }
        return true;
    case GGML_Q4_0:
        for (int64_t b = 0; b < n / 32; ++b, x += 32, o += 18) {
            float amax = 0, mx = 0;
            for (int j = 0; j < 32; ++j)
                if (std::fabs(x[j]) > amax) { amax = std::fabs(x[j]); mx = x[j]; }
            float d = mx / -8.f, id = d != 0 ? 1.f / d : 0.f;
            uint16_t h = f2h(d);
            std::memcpy(o, &h, 2);
            for (int j = 0; j < 16; ++j) {
                int a = std::min(15, (int)(x[j] * id + 8.5f)), c = std::min(15, (int)(x[j + 16] * id + 8.5f));
                o[2 + j] = (uint8_t)(std::max(0, a) | (std::max(0, c) << 4));
            }
        }
        return true;
    default:
        return false;
    }
}

// ------------------------------------------------------------ QMat

void QMat::row(int r, float* out) const { dequant_blocks(type, data + (size_t)r * row_bytes, cols / ggml_block_size(type), out); }

namespace {
struct MulJob {
    const QMat* w;
    float* y;
    const float* b;
};

// Y = X W^T (+ b) for several matrices with the same input, in one parallel loop.
void mul_jobs(const MulJob* jobs, int nj, const float* X, int n) {
    const Kernels& k = cpu_kernels();
    const int RB = 16;
    int start[8] = {0};
    for (int j = 0; j < nj; ++j) start[j + 1] = start[j] + (jobs[j].w->rows + RB - 1) / RB;
    parallel_for(start[nj], [&](int s, int e) {
        std::vector<float> buf;
        int j = 0;
        for (int blk = s; blk < e; ++blk) {
            while (blk >= start[j + 1]) ++j;
            const QMat& m = *jobs[j].w;
            int r0 = (blk - start[j]) * RB, nr = std::min(RB, m.rows - r0), cols = m.cols;
            const float* W;
            if (m.type == GGML_F32) {
                W = (const float*)(m.data + (size_t)r0 * m.row_bytes);
            } else {
                buf.resize((size_t)RB * cols);
                for (int i = 0; i < nr; ++i) m.row(r0 + i, buf.data() + (size_t)i * cols);
                W = buf.data();
            }
            const float* b = jobs[j].b;
            for (int t = 0; t < n; ++t) {
                const float* x = X + (size_t)t * cols;
                float* y = jobs[j].y + (size_t)t * m.rows + r0;
                for (int i = 0; i < nr; ++i) y[i] = k.dot(x, W + (size_t)i * cols, cols) + (b ? b[r0 + i] : 0.f);
            }
        }
    });
}
}  // namespace

void QMat::mul(const float* X, int n, float* Y, const float* b) const {
    MulJob j{this, Y, b};
    mul_jobs(&j, 1, X, n);
}

void QMat::mul_back(const float* dY, int n, float* dX) const {
    const Kernels& k = cpu_kernels();
    int bs = ggml_block_size(type), nblk = cols / bs;
    int step = std::max(1, 64 / bs);  // column slices of at least 64 values
    const int RB = 16;
    parallel_for((nblk + step - 1) / step, [&](int s, int e) {
        int b0 = s * step, b1 = std::min(nblk, e * step);
        int c0 = b0 * bs, len = (b1 - b0) * bs;
        std::vector<float> buf((size_t)RB * len);
        size_t off = (size_t)b0 * ggml_block_bytes(type);
        for (int r0 = 0; r0 < rows; r0 += RB) {
            int nr = std::min(RB, rows - r0);
            for (int i = 0; i < nr; ++i) dequant_blocks(type, data + (size_t)(r0 + i) * row_bytes + off, b1 - b0, buf.data() + (size_t)i * len);
            for (int t = 0; t < n; ++t) {
                const float* dy = dY + (size_t)t * rows + r0;
                float* dx = dX + (size_t)t * cols + c0;
                for (int i = 0; i < nr; ++i)
                    if (dy[i] != 0.f) k.axpy(dy[i], buf.data() + (size_t)i * len, dx, len);
            }
        }
    });
}

// ------------------------------------------------------------ GgufWriter

namespace {
void put_bytes(std::vector<uint8_t>& o, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    o.insert(o.end(), b, b + n);
}
template <class T> void put(std::vector<uint8_t>& o, T v) { put_bytes(o, &v, sizeof v); }
void put_str(std::vector<uint8_t>& o, const std::string& s) {
    put<uint64_t>(o, s.size());
    put_bytes(o, s.data(), s.size());
}
}  // namespace

void GgufWriter::set_str(const std::string& key, const std::string& v) {
    Kv k{key, {}};
    put<uint32_t>(k.bytes, GV_STR);
    put_str(k.bytes, v);
    kvs_.push_back(std::move(k));
}
void GgufWriter::set_u32(const std::string& key, uint32_t v) {
    Kv k{key, {}};
    put<uint32_t>(k.bytes, GV_U32);
    put(k.bytes, v);
    kvs_.push_back(std::move(k));
}
void GgufWriter::set_f32(const std::string& key, float v) {
    Kv k{key, {}};
    put<uint32_t>(k.bytes, GV_F32);
    put(k.bytes, v);
    kvs_.push_back(std::move(k));
}
void GgufWriter::set_bool(const std::string& key, bool v) {
    Kv k{key, {}};
    put<uint32_t>(k.bytes, GV_BOOL);
    put<uint8_t>(k.bytes, v ? 1 : 0);
    kvs_.push_back(std::move(k));
}
void GgufWriter::set_strs(const std::string& key, const std::vector<std::string>& v) {
    Kv k{key, {}};
    put<uint32_t>(k.bytes, GV_ARR);
    put<uint32_t>(k.bytes, GV_STR);
    put<uint64_t>(k.bytes, v.size());
    for (const std::string& s : v) put_str(k.bytes, s);
    kvs_.push_back(std::move(k));
}
void GgufWriter::set_i32s(const std::string& key, const std::vector<int32_t>& v) {
    Kv k{key, {}};
    put<uint32_t>(k.bytes, GV_ARR);
    put<uint32_t>(k.bytes, GV_I32);
    put<uint64_t>(k.bytes, v.size());
    put_bytes(k.bytes, v.data(), v.size() * 4);
    kvs_.push_back(std::move(k));
}
void GgufWriter::set_f32s(const std::string& key, const std::vector<float>& v) {
    Kv k{key, {}};
    put<uint32_t>(k.bytes, GV_ARR);
    put<uint32_t>(k.bytes, GV_F32);
    put<uint64_t>(k.bytes, v.size());
    put_bytes(k.bytes, v.data(), v.size() * 4);
    kvs_.push_back(std::move(k));
}
void GgufWriter::add_tensor(const std::string& name, std::vector<int64_t> dims, int type, std::vector<uint8_t> data) {
    tns_.push_back({name, std::move(dims), type, std::move(data)});
}

bool GgufWriter::save(const std::string& path, std::string* err) const {
    const uint64_t align = 32;
    std::vector<uint8_t> h;
    put_bytes(h, "GGUF", 4);
    put<uint32_t>(h, 3);
    put<uint64_t>(h, tns_.size());
    put<uint64_t>(h, kvs_.size());
    for (const Kv& k : kvs_) {
        put_str(h, k.key);
        put_bytes(h, k.bytes.data(), k.bytes.size());
    }
    uint64_t off = 0;
    for (const Tn& t : tns_) {
        put_str(h, t.name);
        put<uint32_t>(h, (uint32_t)t.dims.size());
        for (int64_t d : t.dims) put<uint64_t>(h, (uint64_t)d);
        put<uint32_t>(h, (uint32_t)t.type);
        put<uint64_t>(h, off);
        off += (t.data.size() + align - 1) / align * align;
    }
    h.resize((h.size() + align - 1) / align * align, 0);
    std::string tmp = path + ".tmp";
    {
        std::ofstream f(std::filesystem::u8path(tmp), std::ios::binary);
        if (!f) {
            if (err) *err = "cannot write " + path;
            return false;
        }
        f.write((const char*)h.data(), (std::streamsize)h.size());
        static const char zeros[32] = {};
        for (const Tn& t : tns_) {
            f.write((const char*)t.data.data(), (std::streamsize)t.data.size());
            size_t pad = (size_t)((t.data.size() + align - 1) / align * align - t.data.size());
            f.write(zeros, (std::streamsize)pad);
        }
        if (!f) {
            if (err) *err = "cannot write " + path;
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(std::filesystem::u8path(tmp), std::filesystem::u8path(path), ec);
    if (ec) {
        std::filesystem::remove(std::filesystem::u8path(path), ec);
        std::filesystem::rename(std::filesystem::u8path(tmp), std::filesystem::u8path(path), ec);
    }
    if (ec && err) *err = "cannot write " + path;
    return !ec;
}

// ------------------------------------------------------------ the model

namespace {
struct Lora {
    Param A, B;  // A [in x r], B [r x out]
};
}  // namespace

struct GgufModel::Impl {
    GgufFile f;
    LlmTokenizer tok;
    int chat = CK_PLAIN;
    std::string sys_default;
    struct Layer {
        std::vector<float> an, fn;
        QMat w[7];                 // q, k, v, o, gate, up, down
        std::vector<float> b[4];   // biases of q, k, v, o (may be empty)
        std::unique_ptr<Lora> lo[7];
        std::vector<float> K, V;  // cache [positions x kv_heads * head_dim]
    };
    std::vector<Layer> L;
    QMat emb, out;
    std::vector<float> onorm, ff;  // final norm, rope frequency factors
    std::vector<int> cached;      // the tokens whose keys and values are in the cache
    std::vector<float> logits;
    std::vector<float> rc, rs;    // rope tables [positions x head_dim / 2]
    int rope_len = 0;
    float lscale = 0;

    // Y = X W^T (+ b) (+ LoRA) for the matrices `which` of a layer that share the
    // input X; keeps X A * scale in tsave[k] when asked.
    void linears(Layer& l, std::initializer_list<int> which, const float* X, int n, std::initializer_list<float*> Y,
                 std::vector<float>* const* tsave = nullptr) {
        MulJob jobs[7] = {};
        int nj = 0;
        auto y = Y.begin();
        for (int w : which) {
            jobs[nj++] = {&l.w[w], *y++, w < 4 && !l.b[w].empty() ? l.b[w].data() : nullptr};
        }
        mul_jobs(jobs, nj, X, n);
        int k = 0;
        y = Y.begin();
        for (int w : which) {
            lora_add(l, w, X, n, *y++, tsave ? tsave[k] : nullptr);
            ++k;
        }
    }
    void linear(Layer& l, int which, const float* X, int n, float* Y, std::vector<float>* tsave = nullptr) {
        std::vector<float>* ts[1] = {tsave};
        linears(l, {which}, X, n, {Y}, ts);
    }
    void lora_add(Layer& l, int which, const float* X, int n, float* Y, std::vector<float>* tsave) {
        const QMat& W = l.w[which];
        Lora* lo = l.lo[which].get();
        if (!lo) return;
        int r = lo->A.w.cols;
        std::vector<float> tmp;
        std::vector<float>& t = tsave ? *tsave : tmp;
        t.assign((size_t)n * r, 0.f);
        const Kernels& k = cpu_kernels();
        k.mm(X, lo->A.w.data(), t.data(), n, W.cols, r);
        for (float& v : t) v *= lscale;
        k.mm(t.data(), lo->B.w.data(), Y, n, r, W.rows);
    }

    // dX += dY W (+ LoRA, whose gradients are added up); ts: the saved X A * scale.
    void linear_back(Layer& l, int which, const float* X, const float* dY, int n, float* dX, const std::vector<float>* ts) {
        const QMat& W = l.w[which];
        W.mul_back(dY, n, dX);
        Lora* lo = l.lo[which].get();
        if (!lo || !ts) return;
        int r = lo->A.w.cols;
        const Kernels& k = cpu_kernels();
        k.mm_ta(ts->data(), dY, lo->B.g.data(), n, r, W.rows);
        std::vector<float> dt((size_t)n * r, 0.f);
        k.mm_tb(dY, lo->B.w.data(), dt.data(), n, W.rows, r);
        for (float& v : dt) v *= lscale;
        k.mm_ta(X, dt.data(), lo->A.g.data(), n, W.cols, r);
        k.mm_tb(dt.data(), lo->A.w.data(), dX, n, r, W.cols);
    }

    void rope_table(int need, const GgufModel::Config& c) {
        if (need <= rope_len) return;
        int hh = c.head_dim / 2, len = std::max(need, rope_len * 2);
        rc.resize((size_t)len * hh);
        rs.resize((size_t)len * hh);
        for (int p = rope_len; p < len; ++p)
            for (int i = 0; i < hh; ++i) {
                double f = std::pow((double)c.rope_base, -2.0 * i / c.head_dim);
                if (!ff.empty()) f /= ff[i];
                double a = p * f;
                rc[(size_t)p * hh + i] = (float)std::cos(a);
                rs[(size_t)p * hh + i] = (float)std::sin(a);
            }
        rope_len = len;
    }

    // Rotates the nh heads of n rows; pos[t] is the position of row t.
    void rope(float* m, int n, int nh, const int* pos, bool inverse, const GgufModel::Config& c) {
        int hd = c.head_dim, hh = hd / 2;
        bool neox = c.rope_neox;
        parallel_for(n, [&](int s, int e) {
            for (int t = s; t < e; ++t) {
                const float* cs = rc.data() + (size_t)pos[t] * hh;
                const float* sn = rs.data() + (size_t)pos[t] * hh;
                for (int h = 0; h < nh; ++h) {
                    float* r = m + ((size_t)t * nh + h) * hd;
                    for (int i = 0; i < hh; ++i) {
                        int a = neox ? i : 2 * i, b = neox ? i + hh : 2 * i + 1;
                        float x0 = r[a], x1 = r[b], si = inverse ? -sn[i] : sn[i];
                        r[a] = x0 * cs[i] - x1 * si;
                        r[b] = x0 * si + x1 * cs[i];
                    }
                }
            }
        }, 4);
    }
};

namespace {
void rmsnorm(const float* x, const float* g, float* y, float* inv, int n, int C, float eps) {
    parallel_for(n, [&](int s, int e) {
        for (int t = s; t < e; ++t) {
            const float* a = x + (size_t)t * C;
            double ss = 0;
            for (int j = 0; j < C; ++j) ss += (double)a[j] * a[j];
            float r = (float)(1.0 / std::sqrt(ss / C + eps));
            if (inv) inv[t] = r;
            float* o = y + (size_t)t * C;
            for (int j = 0; j < C; ++j) o[j] = a[j] * r * g[j];
        }
    }, 8);
}

// dx += d(rmsnorm)/dx * dy, with r = 1 / rms of each row.
void rmsnorm_back(const float* x, const float* g, const float* r, const float* dy, float* dx, int n, int C) {
    parallel_for(n, [&](int s, int e) {
        for (int t = s; t < e; ++t) {
            const float* a = x + (size_t)t * C;
            const float* d = dy + (size_t)t * C;
            float* o = dx + (size_t)t * C;
            double m = 0;
            for (int j = 0; j < C; ++j) m += (double)d[j] * g[j] * a[j] * r[t];
            float mf = (float)(m / C);
            for (int j = 0; j < C; ++j) o[j] += r[t] * (d[j] * g[j] - a[j] * r[t] * mf);
        }
    }, 8);
}

inline float silu(float x) { return x / (1.f + std::exp(-x)); }
}  // namespace

GgufModel::GgufModel() : d(new Impl) {}
GgufModel::~GgufModel() = default;

std::shared_ptr<GgufModel> GgufModel::load(const std::string& path, std::string* err, int max_ctx) {
    std::shared_ptr<GgufModel> m(new GgufModel());
    Impl& d = *m->d;
    auto fail = [&](const std::string& msg) -> std::shared_ptr<GgufModel> {
        if (err) *err = msg;
        return nullptr;
    };
    if (!d.f.open(path, err)) return nullptr;
    const GgufFile& f = d.f;
    m->path = path;
    m->file_bytes = f.f.size;
    Config& c = m->cfg;
    c.arch = f.str("general.architecture");
    if (c.arch != "llama" && c.arch != "qwen2")
        return fail("unsupported model architecture '" + c.arch + "' (supported: llama, mistral, qwen2)");
    m->name = f.str("general.name", std::filesystem::u8path(path).stem().u8string());
    std::string a = c.arch + ".";
    c.dim = (int)f.num(a + "embedding_length", 0);
    c.layers = (int)f.num(a + "block_count", 0);
    c.ffn = (int)f.num(a + "feed_forward_length", 0);
    c.heads = (int)f.num(a + "attention.head_count", 0);
    c.kv_heads = (int)f.num(a + "attention.head_count_kv", c.heads);
    c.ctx_train = (int)f.num(a + "context_length", 2048);
    c.rope_base = (float)f.num(a + "rope.freq_base", 10000.0);
    c.eps = (float)f.num(a + "attention.layer_norm_rms_epsilon", 1e-5);
    c.head_dim = c.heads > 0 ? (int)f.num(a + "attention.key_length", c.dim / c.heads) : 0;
    c.rope_neox = c.arch == "qwen2";
    if (c.dim <= 0 || c.layers <= 0 || c.ffn <= 0 || c.heads <= 0 || c.kv_heads <= 0 || c.heads % c.kv_heads ||
        c.head_dim <= 0 || c.head_dim % 2)
        return fail("the model settings in the file are not valid");
    if (f.get(a + "rope.scaling.type") && f.str(a + "rope.scaling.type") != "none" && f.str(a + "rope.scaling.type") != "")
        return fail("rope scaling '" + f.str(a + "rope.scaling.type") + "' is not supported");

    std::string bad;
    auto mat = [&](const std::string& name, int rows, int cols, QMat& q) {
        const GgufTensor* t = f.tensor(name);
        if (!t) { if (bad.empty()) bad = "missing tensor " + name; return false; }
        if (!type_info(t->type)) { if (bad.empty()) bad = name + " uses an unsupported quantization (type " + std::to_string(t->type) + ")"; return false; }
        if (t->dims.size() != 2 || t->dims[0] != cols || t->dims[1] != rows || !t->data) {
            if (bad.empty()) bad = "tensor " + name + " has the wrong shape";
            return false;
        }
        q.type = t->type;
        q.rows = rows;
        q.cols = cols;
        q.data = t->data;
        q.row_bytes = ggml_row_bytes(t->type, cols);
        return true;
    };
    auto vec = [&](const std::string& name, int n, std::vector<float>& v, bool need) {
        const GgufTensor* t = f.tensor(name);
        if (!t) { if (need && bad.empty()) bad = "missing tensor " + name; return !need; }
        if (!type_info(t->type) || t->count() != n || !t->data) { if (bad.empty()) bad = "tensor " + name + " is not valid"; return false; }
        v.resize(n);
        ggml_dequantize(t->type, t->data, n, v.data());
        return true;
    };

    const GgufTensor* te = f.tensor("token_embd.weight");
    if (!te || te->dims.size() != 2 || te->dims[0] != c.dim) return fail("missing or invalid token_embd.weight");
    c.vocab = (int)te->dims[1];
    int C = c.dim, qd = c.heads * c.head_dim, kvd = c.kv_heads * c.head_dim;
    bool ok = mat("token_embd.weight", c.vocab, C, d.emb) && vec("output_norm.weight", C, d.onorm, true);
    if (f.tensor("output.weight")) ok = ok && mat("output.weight", c.vocab, C, d.out);
    else d.out = d.emb;
    vec("rope_freqs.weight", c.head_dim / 2, d.ff, false);
    d.L.resize(c.layers);
    for (int l = 0; l < c.layers && ok; ++l) {
        Impl::Layer& y = d.L[l];
        std::string p = "blk." + std::to_string(l) + ".";
        ok = vec(p + "attn_norm.weight", C, y.an, true) && vec(p + "ffn_norm.weight", C, y.fn, true) &&
             mat(p + "attn_q.weight", qd, C, y.w[0]) && mat(p + "attn_k.weight", kvd, C, y.w[1]) &&
             mat(p + "attn_v.weight", kvd, C, y.w[2]) && mat(p + "attn_output.weight", C, qd, y.w[3]) &&
             mat(p + "ffn_gate.weight", c.ffn, C, y.w[4]) && mat(p + "ffn_up.weight", c.ffn, C, y.w[5]) &&
             mat(p + "ffn_down.weight", C, c.ffn, y.w[6]) && vec(p + "attn_q.bias", qd, y.b[0], false) &&
             vec(p + "attn_k.bias", kvd, y.b[1], false) && vec(p + "attn_v.bias", kvd, y.b[2], false) &&
             vec(p + "attn_output.bias", C, y.b[3], false);
        if (ok && f.tensor(p + "ffn_gate_inp.weight")) { bad = "mixture-of-experts models are not supported"; ok = false; }
    }
    if (!ok) return fail(bad.empty() ? "the model file is not valid" : bad);

    std::map<int, size_t> by_type;
    for (const auto& [n, t] : f.tensors) {
        m->n_params += (size_t)t.count();
        if (t.dims.size() == 2) by_type[t.type] += (size_t)t.count();
    }
    int top = GGML_F32;
    size_t best = 0;
    for (auto& [t, n] : by_type)
        if (n > best) { best = n; top = t; }
    m->quant = ggml_type_name(top) ? ggml_type_name(top) : "?";
    size_t total2d = 0;
    for (auto& [t, n] : by_type) total2d += n;
    for (auto& [t, n] : by_type)
        m->quant_mix += (m->quant_mix.empty() ? "" : ", ") + std::string(ggml_type_name(t) ? ggml_type_name(t) : "?") + " " +
                        std::to_string((int)std::lround(100.0 * n / std::max<size_t>(1, total2d))) + "%";

    // ---- tokenizer
    LlmTokenizer& tk = d.tok;
    std::string tm = f.str("tokenizer.ggml.model");
    if (tm != "llama" && tm != "gpt2") return fail("unsupported tokenizer '" + tm + "'");
    tk.spm = tm == "llama";
    const GgufKv* toks = f.get("tokenizer.ggml.tokens");
    if (!toks || toks->strs.empty()) return fail("the file has no tokenizer");
    tk.toks = toks->strs;
    if ((int)tk.toks.size() > c.vocab) tk.toks.resize(c.vocab);
    int nt = (int)tk.toks.size();
    tk.toks.resize(c.vocab);
    tk.types.assign(c.vocab, TT_UNUSED);
    tk.scores.assign(c.vocab, 0.f);
    const GgufKv* ty = f.get("tokenizer.ggml.token_type");
    const GgufKv* sc = f.get("tokenizer.ggml.scores");
    for (int i = 0; i < nt; ++i) {
        tk.types[i] = ty && i < (int)ty->nums.size() ? (int)ty->nums[i] : TT_NORMAL;
        if (sc && i < (int)sc->nums.size()) tk.scores[i] = (float)sc->nums[i];
    }
    for (int i = 0; i < nt; ++i)
        if (tk.types[i] != TT_UNUSED || tk.index.find(tk.toks[i]) == tk.index.end()) tk.index.emplace(tk.toks[i], i);
    if (!tk.spm) {
        tk.init_bytes();
        if (const GgufKv* mg = f.get("tokenizer.ggml.merges"))
            for (size_t i = 0; i < mg->strs.size(); ++i) tk.merges.emplace(mg->strs[i], (int)i);
        std::string pre = f.str("tokenizer.ggml.pre", "default");
        if (pre == "qwen2" || pre == "deepseek-r1-qwen") { tk.llama3_split = true; tk.digits = 1; }
        else if (pre == "llama3" || pre == "llama-bpe" || pre == "smaug-bpe" || pre == "dbrx") { tk.llama3_split = true; tk.digits = 3; }
        else if (pre == "smollm") { tk.digits = 1; }
    }
    for (int b = 0; b < 256; ++b) {
        char name[8];
        std::snprintf(name, sizeof name, "<0x%02X>", b);
        tk.byte_tok[b] = tk.find(name);
    }
    tk.bos = (int)f.num("tokenizer.ggml.bos_token_id", -1);
    tk.eos = (int)f.num("tokenizer.ggml.eos_token_id", -1);
    tk.unk = (int)f.num("tokenizer.ggml.unknown_token_id", -1);
    if (tk.bos >= c.vocab) tk.bos = -1;
    if (tk.eos >= c.vocab) tk.eos = -1;
    if (tk.unk >= c.vocab) tk.unk = -1;
    tk.add_bos = f.num("tokenizer.ggml.add_bos_token", tk.spm ? 1 : 0) != 0;
    tk.add_space = f.num("tokenizer.ggml.add_space_prefix", 1) != 0;
    tk.special_by_byte.assign(256, {});
    for (int i = 0; i < nt; ++i)
        if ((tk.types[i] == TT_CONTROL || tk.types[i] == TT_USER) && !tk.toks[i].empty())
            tk.special_by_byte[(unsigned char)tk.toks[i][0]].push_back(i);
    for (auto& v : tk.special_by_byte)
        std::sort(v.begin(), v.end(), [&](int x, int y) { return tk.toks[x].size() > tk.toks[y].size(); });
    tk.end.assign(c.vocab, 0);
    for (const char* key : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"}) {
        int id = (int)f.num(key, -1);
        if (id >= 0 && id < c.vocab) tk.end[id] = 1;
    }
    for (const char* s : {"<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|end|>", "<|endoftext|>", "</s>", "<|end_of_text|>", "<end_of_turn>"}) {
        int id = tk.find(s);
        if (id >= 0 && tk.types[id] != TT_NORMAL) tk.end[id] = 1;
        else if (id >= 0 && id == tk.eos) tk.end[id] = 1;
    }

    // ---- chat format
    std::string tmpl = f.str("tokenizer.chat_template");
    if (tmpl.find("<|im_start|>") != std::string::npos) d.chat = CK_CHATML;
    else if (tmpl.find("<|start_header_id|>") != std::string::npos) d.chat = CK_LLAMA3;
    else if (tmpl.find("<start_of_turn>") != std::string::npos) d.chat = CK_GEMMA;
    else if (tmpl.find("<|user|>") != std::string::npos && tmpl.find("<|end|>") != std::string::npos) d.chat = CK_PHI3;
    else if (tmpl.find("<|user|>") != std::string::npos) d.chat = CK_ZEPHYR;
    else if (tmpl.find("[INST]") != std::string::npos) d.chat = CK_MISTRAL;
    else if (tk.find("<|im_start|>") >= 0) d.chat = CK_CHATML;
    const char* qwen_sys = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    if (tmpl.find(qwen_sys) != std::string::npos) d.sys_default = qwen_sys;

    m->ctx = std::max(64, std::min(max_ctx, c.ctx_train > 0 ? c.ctx_train : max_ctx));
    size_t per_pos = (size_t)c.layers * kvd * 2 * sizeof(float);
    m->ctx = std::max(64, std::min(m->ctx, (int)std::min<size_t>(INT32_MAX, ((size_t)1 << 30) / per_pos)));
    return m;
}

std::vector<int> GgufModel::encode(const std::string& text, bool add_bos, bool special) const {
    return d->tok.encode(text, add_bos && d->tok.add_bos, special);
}
std::string GgufModel::piece(int id) const { return d->tok.piece(id); }
std::string GgufModel::decode(const std::vector<int>& ids) const {
    std::string s;
    for (int id : ids) s += d->tok.piece(id);
    return s;
}
bool GgufModel::is_end(int id) const { return id >= 0 && id < (int)d->tok.end.size() && d->tok.end[id]; }
int GgufModel::eos() const { return d->tok.eos; }
std::string GgufModel::chat_format() const { return kChatNames[d->chat]; }

std::string GgufModel::chat_text(const std::vector<LlmMsg>& msgs, bool open_answer) const {
    std::string sys;
    std::vector<const LlmMsg*> turns;
    for (const LlmMsg& m : msgs) {
        if (m.role == "system") sys += (sys.empty() ? "" : "\n") + m.text;
        else turns.push_back(&m);
    }
    if (sys.empty()) sys = d->sys_default;
    std::string o;
    auto role = [](const LlmMsg* m) { return m->role == "assistant" ? std::string("assistant") : std::string("user"); };
    switch (d->chat) {
    case CK_CHATML:
        if (!sys.empty()) o += "<|im_start|>system\n" + sys + "<|im_end|>\n";
        for (auto* m : turns) o += "<|im_start|>" + role(m) + "\n" + m->text + "<|im_end|>\n";
        if (open_answer) o += "<|im_start|>assistant\n";
        break;
    case CK_LLAMA3:
        if (!sys.empty()) o += "<|start_header_id|>system<|end_header_id|>\n\n" + trim(sys) + "<|eot_id|>";
        for (auto* m : turns) o += "<|start_header_id|>" + role(m) + "<|end_header_id|>\n\n" + trim(m->text) + "<|eot_id|>";
        if (open_answer) o += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        break;
    case CK_MISTRAL: {
        bool first = true;
        for (auto* m : turns) {
            if (role(m) == "user") {
                o += "[INST] " + (first && !sys.empty() ? sys + "\n\n" : std::string()) + m->text + " [/INST]";
                first = false;
            } else {
                o += " " + m->text + "</s>";
            }
        }
        break;
    }
    case CK_PHI3:
        if (!sys.empty()) o += "<|system|>\n" + sys + "<|end|>\n";
        for (auto* m : turns) o += "<|" + role(m) + "|>\n" + m->text + "<|end|>\n";
        if (open_answer) o += "<|assistant|>\n";
        break;
    case CK_ZEPHYR:
        if (!sys.empty()) o += "<|system|>\n" + sys + "</s>\n";
        for (auto* m : turns) o += "<|" + role(m) + "|>\n" + m->text + "</s>\n";
        if (open_answer) o += "<|assistant|>\n";
        break;
    case CK_GEMMA: {
        bool first = true;
        for (auto* m : turns) {
            bool user = role(m) == "user";
            o += std::string("<start_of_turn>") + (user ? "user" : "model") + "\n" +
                 (user && first && !sys.empty() ? sys + "\n\n" : std::string()) + m->text + "<end_of_turn>\n";
            if (user) first = false;
        }
        if (open_answer) o += "<start_of_turn>model\n";
        break;
    }
    default:
        if (!sys.empty()) o += sys + "\n\n";
        for (auto* m : turns) o += (role(m) == "user" ? "User: " : "Assistant: ") + m->text + "\n";
        if (open_answer) o += "Assistant:";
    }
    return o;
}

std::vector<int> GgufModel::chat_ids(const std::vector<LlmMsg>& msgs, bool open_answer) const {
    return encode(chat_text(msgs, open_answer), true, true);
}

const float* GgufModel::eval(const int* toks, int n, int pos0) {
    Impl& m = *d;
    const Config& c = cfg;
    if (n <= 0 || pos0 < 0 || pos0 + n > ctx) return nullptr;
    for (int t = 0; t < n; ++t)
        if (toks[t] < 0 || toks[t] >= c.vocab) return nullptr;
    int C = c.dim, H = c.heads, KV = c.kv_heads, hd = c.head_dim, F = c.ffn, qd = H * hd, kvd = KV * hd, grp = H / KV;
    m.rope_table(pos0 + n, c);
    std::vector<float> x((size_t)n * C), h((size_t)n * C), q((size_t)n * qd), k((size_t)n * kvd), v((size_t)n * kvd),
        ao((size_t)n * qd), o((size_t)n * C), g((size_t)n * F), u((size_t)n * F);
    for (int t = 0; t < n; ++t) m.emb.row(toks[t], x.data() + (size_t)t * C);
    std::vector<int> pos(n);
    for (int t = 0; t < n; ++t) pos[t] = pos0 + t;
    size_t need = (size_t)(pos0 + n) * kvd;
    float scale = 1.f / std::sqrt((float)hd);
    const Kernels& kk = cpu_kernels();
    for (Impl::Layer& l : m.L) {
        if (l.K.size() < need) {
            size_t cap = std::min((size_t)ctx * kvd, std::max(need, l.K.size() * 2));
            l.K.resize(cap);
            l.V.resize(cap);
        }
        rmsnorm(x.data(), l.an.data(), h.data(), nullptr, n, C, c.eps);
        m.linears(l, {0, 1, 2}, h.data(), n, {q.data(), k.data(), v.data()});
        m.rope(q.data(), n, H, pos.data(), false, c);
        m.rope(k.data(), n, KV, pos.data(), false, c);
        std::memcpy(l.K.data() + (size_t)pos0 * kvd, k.data(), sizeof(float) * n * kvd);
        std::memcpy(l.V.data() + (size_t)pos0 * kvd, v.data(), sizeof(float) * n * kvd);
        parallel_for(n * H, [&](int s, int e) {
            std::vector<float> sc(pos0 + n);
            for (int idx = s; idx < e; ++idx) {
                int t = idx / H, hh = idx % H, gk = hh / grp, last = pos0 + t;
                const float* qt = q.data() + (size_t)t * qd + hh * hd;
                float mx = -1e30f;
                for (int j = 0; j <= last; ++j) {
                    sc[j] = kk.dot(qt, l.K.data() + (size_t)j * kvd + gk * hd, hd) * scale;
                    mx = std::max(mx, sc[j]);
                }
                double sum = 0;
                for (int j = 0; j <= last; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                float* out = ao.data() + (size_t)t * qd + hh * hd;
                std::fill(out, out + hd, 0.f);
                float is = (float)(1.0 / sum);
                for (int j = 0; j <= last; ++j) kk.axpy(sc[j] * is, l.V.data() + (size_t)j * kvd + gk * hd, out, hd);
            }
        });
        m.linear(l, 3, ao.data(), n, o.data());
        for (size_t i = 0; i < x.size(); ++i) x[i] += o[i];
        rmsnorm(x.data(), l.fn.data(), h.data(), nullptr, n, C, c.eps);
        m.linears(l, {4, 5}, h.data(), n, {g.data(), u.data()});
        for (size_t i = 0; i < g.size(); ++i) g[i] = silu(g[i]) * u[i];
        m.linear(l, 6, g.data(), n, o.data());
        for (size_t i = 0; i < x.size(); ++i) x[i] += o[i];
    }
    rmsnorm(x.data() + (size_t)(n - 1) * C, m.onorm.data(), h.data(), nullptr, 1, C, c.eps);
    m.logits.resize(c.vocab);
    m.out.mul(h.data(), 1, m.logits.data());
    m.cached.resize(std::min((int)m.cached.size(), pos0));
    m.cached.insert(m.cached.end(), toks, toks + n);
    return m.logits.data();
}

int GgufModel::generate(const std::vector<int>& prompt0, int n, const Sampling& s, Rng& rng,
                        const std::function<bool(int)>& on_token) {
    Impl& m = *d;
    if (prompt0.empty() || n <= 0) return 0;
    std::vector<int> prompt = prompt0;
    int keep = ctx - std::min(n, ctx / 2);
    if ((int)prompt.size() > keep) prompt.erase(prompt.begin(), prompt.end() - keep);
    size_t same = 0;
    while (same < m.cached.size() && same < prompt.size() && m.cached[same] == prompt[same]) ++same;
    if (same == prompt.size()) --same;
    const float* z = nullptr;
    for (size_t i = same; i < prompt.size(); i += 256) {
        int k = (int)std::min<size_t>(256, prompt.size() - i);
        z = eval(prompt.data() + i, k, (int)i);
        if (!z) return 0;
    }
    std::vector<int> recent = prompt;
    int made = 0;
    while (made < n) {
        int t = sample_logits(z, cfg.vocab, s, recent.data(), recent.size(), rng, eos());
        if (is_end(t)) break;
        ++made;
        recent.push_back(t);
        if (!on_token(t)) break;
        if ((int)m.cached.size() >= ctx || made >= n) break;
        z = eval(&t, 1, (int)m.cached.size());
        if (!z) break;
    }
    return made;
}

std::vector<int> GgufModel::chat_prompt(const std::vector<LlmMsg>& msgs0, int room) const {
    std::vector<LlmMsg> msgs = msgs0;
    int limit = std::max(16, ctx - std::max(1, std::min(room, ctx / 2)));
    std::vector<int> ids = chat_ids(msgs);
    while ((int)ids.size() > limit) {
        size_t first = 0;
        while (first < msgs.size() && msgs[first].role == "system") ++first;
        if (first + 1 >= msgs.size()) break;
        msgs.erase(msgs.begin() + first);
        ids = chat_ids(msgs);
    }
    if ((int)ids.size() > limit) ids.erase(ids.begin(), ids.end() - limit);
    return ids;
}

std::string GgufModel::reply(const std::vector<int>& prompt, int n, const Sampling& s, Rng& rng,
                             const std::function<bool(const std::string&)>& on_text, int* tokens) {
    std::string pending, all;
    int made = generate(prompt, n, s, rng, [&](int t) {
        pending += piece(t);
        size_t k = utf8_whole(pending);
        if (k == 0) return true;
        std::string part = pending.substr(0, k);
        pending.erase(0, k);
        all += part;
        return on_text(part);
    });
    if (!pending.empty()) {
        all += pending;
        on_text(pending);
    }
    if (tokens) *tokens = made;
    return all;
}

// ------------------------------------------------------------ LoRA

bool GgufModel::has_lora() const {
    for (const auto& l : d->L)
        for (const auto& lo : l.lo)
            if (lo) return true;
    return false;
}

void GgufModel::lora_clear() {
    for (auto& l : d->L)
        for (auto& lo : l.lo) lo.reset();
    d->cached.clear();
}

void GgufModel::lora_init(const LoraConfig& c) {
    lora_clear();
    lora_cfg = c;
    lora_cfg.rank = std::max(1, c.rank);
    int r = lora_cfg.rank;
    d->lscale = lora_cfg.alpha / r;
    Rng rng(c.seed * 7919 + 13);
    for (size_t li = 0; li < d->L.size(); ++li) {
        Impl::Layer& l = d->L[li];
        for (int w = 0; w < 7; ++w) {
            if (w >= 4 && !c.mlp) continue;
            auto lo = std::make_unique<Lora>();
            int in = l.w[w].cols, out = l.w[w].rows;
            std::string base = "blk." + std::to_string(li) + "." + kLoraNames[w] + ".weight";
            lo->A.init(base + ".lora_a", in, r, false);
            lo->B.init(base + ".lora_b", r, out, false);
            float sd = 1.f / std::sqrt((float)in);
            for (float& v : lo->A.w.v) v = rng.normal() * sd;
            l.lo[w] = std::move(lo);
        }
    }
}

std::vector<Param*> GgufModel::lora_params() {
    std::vector<Param*> ps;
    for (auto& l : d->L)
        for (auto& lo : l.lo)
            if (lo) {
                ps.push_back(&lo->A);
                ps.push_back(&lo->B);
            }
    return ps;
}

float GgufModel::lora_loss(const int* ids, const int* targets, const float* w, int B, int T, bool backward) {
    Impl& m = *d;
    const Config& c = cfg;
    int n = B * T, C = c.dim, H = c.heads, KV = c.kv_heads, hd = c.head_dim, F = c.ffn, qd = H * hd, kvd = KV * hd,
        grp = H / KV, V = c.vocab;
    m.cached.clear();
    m.rope_table(T, c);
    std::vector<int> pos(n);
    for (int i = 0; i < n; ++i) pos[i] = i % T;
    float scale = 1.f / std::sqrt((float)hd);
    const Kernels& kk = cpu_kernels();

    // One layer's intermediates, recomputed in the backward pass.
    struct Acts {
        std::vector<float> h1, r1, q, k, v, P, ao, x2, h2, r2, g, u, a;
        std::vector<float> t[7];
    } A;
    auto layer_fwd = [&](Impl::Layer& l, const float* x, float* y, bool keep) {
        A.h1.resize((size_t)n * C);
        A.r1.resize(n);
        A.q.resize((size_t)n * qd);
        A.k.resize((size_t)n * kvd);
        A.v.resize((size_t)n * kvd);
        A.ao.assign((size_t)n * qd, 0.f);
        A.P.resize((size_t)B * H * T * T);
        rmsnorm(x, l.an.data(), A.h1.data(), A.r1.data(), n, C, c.eps);
        std::vector<float>* tq[3] = {keep ? &A.t[0] : nullptr, keep ? &A.t[1] : nullptr, keep ? &A.t[2] : nullptr};
        m.linears(l, {0, 1, 2}, A.h1.data(), n, {A.q.data(), A.k.data(), A.v.data()}, tq);
        m.rope(A.q.data(), n, H, pos.data(), false, c);
        m.rope(A.k.data(), n, KV, pos.data(), false, c);
        parallel_for(B * H, [&](int s, int e) {
            for (int idx = s; idx < e; ++idx) {
                int b = idx / H, hh = idx % H, gk = hh / grp;
                for (int t = 0; t < T; ++t) {
                    const float* qt = A.q.data() + (size_t)(b * T + t) * qd + hh * hd;
                    float* p = A.P.data() + ((size_t)idx * T + t) * T;
                    float mx = -1e30f;
                    for (int j = 0; j <= t; ++j) {
                        p[j] = kk.dot(qt, A.k.data() + (size_t)(b * T + j) * kvd + gk * hd, hd) * scale;
                        mx = std::max(mx, p[j]);
                    }
                    double sum = 0;
                    for (int j = 0; j <= t; ++j) { p[j] = std::exp(p[j] - mx); sum += p[j]; }
                    float is = (float)(1.0 / sum);
                    float* o = A.ao.data() + (size_t)(b * T + t) * qd + hh * hd;
                    for (int j = 0; j <= t; ++j) {
                        p[j] *= is;
                        kk.axpy(p[j], A.v.data() + (size_t)(b * T + j) * kvd + gk * hd, o, hd);
                    }
                    for (int j = t + 1; j < T; ++j) p[j] = 0.f;
                }
            }
        });
        A.x2.resize((size_t)n * C);
        m.linear(l, 3, A.ao.data(), n, A.x2.data(), keep ? &A.t[3] : nullptr);
        for (size_t i = 0; i < A.x2.size(); ++i) A.x2[i] += x[i];
        A.h2.resize((size_t)n * C);
        A.r2.resize(n);
        rmsnorm(A.x2.data(), l.fn.data(), A.h2.data(), A.r2.data(), n, C, c.eps);
        A.g.resize((size_t)n * F);
        A.u.resize((size_t)n * F);
        A.a.resize((size_t)n * F);
        std::vector<float>* tg[2] = {keep ? &A.t[4] : nullptr, keep ? &A.t[5] : nullptr};
        m.linears(l, {4, 5}, A.h2.data(), n, {A.g.data(), A.u.data()}, tg);
        for (size_t i = 0; i < A.a.size(); ++i) A.a[i] = silu(A.g[i]) * A.u[i];
        m.linear(l, 6, A.a.data(), n, y, keep ? &A.t[6] : nullptr);
        for (size_t i = 0; i < (size_t)n * C; ++i) y[i] += A.x2[i];
    };

    // ---- forward, keeping only each layer's input
    int nl = c.layers;
    std::vector<std::vector<float>> xs(nl + 1, std::vector<float>((size_t)n * C));
    for (int t = 0; t < n; ++t) m.emb.row(ids[t], xs[0].data() + (size_t)t * C);
    for (int l = 0; l < nl; ++l) layer_fwd(m.L[l], xs[l].data(), xs[l + 1].data(), false);

    // ---- head: logits in chunks of rows, only where a target counts
    std::vector<float> hn((size_t)n * C), rf(n), dhn(backward ? (size_t)n * C : 0, 0.f);
    rmsnorm(xs[nl].data(), m.onorm.data(), hn.data(), rf.data(), n, C, c.eps);
    std::vector<int> rows;
    double wsum = 0;
    for (int i = 0; i < n; ++i)
        if (w[i] > 0 && targets[i] >= 0 && targets[i] < V) { rows.push_back(i); wsum += w[i]; }
    if (rows.empty() || wsum <= 0) return 0.f;
    int chunk = std::max(8, std::min(256, (int)(32000000 / V)));
    std::vector<float> X, Z, dX;
    double loss = 0;
    for (size_t r0 = 0; r0 < rows.size(); r0 += chunk) {
        int cn = (int)std::min<size_t>(chunk, rows.size() - r0);
        X.resize((size_t)cn * C);
        Z.resize((size_t)cn * V);
        for (int i = 0; i < cn; ++i) std::memcpy(X.data() + (size_t)i * C, hn.data() + (size_t)rows[r0 + i] * C, sizeof(float) * C);
        m.out.mul(X.data(), cn, Z.data());
        std::vector<double> part(cn, 0.0);
        parallel_for(cn, [&](int s, int e) {
            for (int i = s; i < e; ++i) {
                int row = rows[r0 + i];
                float* z = Z.data() + (size_t)i * V;
                float mx = *std::max_element(z, z + V);
                double sum = 0;
                for (int j = 0; j < V; ++j) sum += std::exp((double)(z[j] - mx));
                double lse = mx + std::log(sum);
                part[i] = w[row] * (lse - z[targets[row]]);
                if (backward) {
                    float k = (float)(w[row] / wsum);
                    for (int j = 0; j < V; ++j) z[j] = k * (float)std::exp(z[j] - lse);
                    z[targets[row]] -= k;
                }
            }
        });
        for (double p : part) loss += p;
        if (backward) {
            dX.assign((size_t)cn * C, 0.f);
            m.out.mul_back(Z.data(), cn, dX.data());
            for (int i = 0; i < cn; ++i)
                std::memcpy(dhn.data() + (size_t)rows[r0 + i] * C, dX.data() + (size_t)i * C, sizeof(float) * C);
        }
    }
    if (!backward) return (float)(loss / wsum);

    // ---- backward, layer by layer, recomputing each layer's forward
    std::vector<float> dx((size_t)n * C, 0.f);
    rmsnorm_back(xs[nl].data(), m.onorm.data(), rf.data(), dhn.data(), dx.data(), n, C);
    std::vector<float> dx2, da, dg, du, dh, dao, dq, dk, dv, tmp((size_t)n * C);
    for (int l = nl - 1; l >= 0; --l) {
        Impl::Layer& L = m.L[l];
        layer_fwd(L, xs[l].data(), tmp.data(), true);
        xs[l + 1].clear();
        xs[l + 1].shrink_to_fit();
        // feed-forward
        dx2 = dx;
        da.assign((size_t)n * F, 0.f);
        m.linear_back(L, 6, A.a.data(), dx.data(), n, da.data(), L.lo[6] ? &A.t[6] : nullptr);
        dg.resize((size_t)n * F);
        du.resize((size_t)n * F);
        for (size_t i = 0; i < da.size(); ++i) {
            float gv = A.g[i], sg = 1.f / (1.f + std::exp(-gv));
            du[i] = da[i] * gv * sg;
            dg[i] = da[i] * A.u[i] * sg * (1.f + gv * (1.f - sg));
        }
        dh.assign((size_t)n * C, 0.f);
        m.linear_back(L, 4, A.h2.data(), dg.data(), n, dh.data(), L.lo[4] ? &A.t[4] : nullptr);
        m.linear_back(L, 5, A.h2.data(), du.data(), n, dh.data(), L.lo[5] ? &A.t[5] : nullptr);
        rmsnorm_back(A.x2.data(), L.fn.data(), A.r2.data(), dh.data(), dx2.data(), n, C);
        // attention
        dao.assign((size_t)n * qd, 0.f);
        m.linear_back(L, 3, A.ao.data(), dx2.data(), n, dao.data(), L.lo[3] ? &A.t[3] : nullptr);
        dq.assign((size_t)n * qd, 0.f);
        dk.assign((size_t)n * kvd, 0.f);
        dv.assign((size_t)n * kvd, 0.f);
        parallel_for(B * KV, [&](int s, int e) {
            std::vector<float> dp(T);
            for (int idx = s; idx < e; ++idx) {
                int b = idx / KV, gk = idx % KV;
                for (int hh = gk * grp; hh < (gk + 1) * grp; ++hh)
                    for (int t = 0; t < T; ++t) {
                        const float* p = A.P.data() + (((size_t)b * H + hh) * T + t) * T;
                        const float* go = dao.data() + (size_t)(b * T + t) * qd + hh * hd;
                        double dot = 0;
                        for (int j = 0; j <= t; ++j) {
                            dp[j] = kk.dot(go, A.v.data() + (size_t)(b * T + j) * kvd + gk * hd, hd);
                            dot += (double)p[j] * dp[j];
                        }
                        const float* qt = A.q.data() + (size_t)(b * T + t) * qd + hh * hd;
                        float* gq = dq.data() + (size_t)(b * T + t) * qd + hh * hd;
                        for (int j = 0; j <= t; ++j) {
                            float ds = p[j] * (dp[j] - (float)dot) * scale;
                            size_t kj = (size_t)(b * T + j) * kvd + gk * hd;
                            kk.axpy(ds, A.k.data() + kj, gq, hd);
                            kk.axpy(ds, qt, dk.data() + kj, hd);
                            kk.axpy(p[j], go, dv.data() + kj, hd);
                        }
                    }
            }
        });
        m.rope(dq.data(), n, H, pos.data(), true, c);
        m.rope(dk.data(), n, KV, pos.data(), true, c);
        dh.assign((size_t)n * C, 0.f);
        m.linear_back(L, 0, A.h1.data(), dq.data(), n, dh.data(), L.lo[0] ? &A.t[0] : nullptr);
        m.linear_back(L, 1, A.h1.data(), dk.data(), n, dh.data(), L.lo[1] ? &A.t[1] : nullptr);
        m.linear_back(L, 2, A.h1.data(), dv.data(), n, dh.data(), L.lo[2] ? &A.t[2] : nullptr);
        dx = dx2;
        rmsnorm_back(xs[l].data(), L.an.data(), A.r1.data(), dh.data(), dx.data(), n, C);
    }
    return (float)(loss / wsum);
}

bool GgufModel::lora_save(const std::string& path, std::string* err) const {
    if (!has_lora()) {
        if (err) *err = "no LoRA adapter to save";
        return false;
    }
    GgufWriter w;
    w.set_str("general.architecture", cfg.arch);
    w.set_str("general.type", "adapter");
    w.set_str("general.name", name + " LoRA");
    w.set_str("adapter.type", "lora");
    w.set_f32("adapter.lora.alpha", lora_cfg.alpha);
    for (size_t li = 0; li < d->L.size(); ++li)
        for (int k = 0; k < 7; ++k) {
            const Lora* lo = d->L[li].lo[k].get();
            if (!lo) continue;
            int in = lo->A.w.rows, r = lo->A.w.cols, out = lo->B.w.cols;
            std::vector<float> at((size_t)r * in), bt((size_t)out * r);
            for (int i = 0; i < in; ++i)
                for (int j = 0; j < r; ++j) at[(size_t)j * in + i] = lo->A.w.v[(size_t)i * r + j];
            for (int i = 0; i < r; ++i)
                for (int j = 0; j < out; ++j) bt[(size_t)j * r + i] = lo->B.w.v[(size_t)i * out + j];
            std::vector<uint8_t> ab, bb;
            ggml_quantize(GGML_F32, at.data(), (int64_t)at.size(), ab);
            ggml_quantize(GGML_F32, bt.data(), (int64_t)bt.size(), bb);
            w.add_tensor(lo->A.name, {in, r}, GGML_F32, std::move(ab));
            w.add_tensor(lo->B.name, {r, out}, GGML_F32, std::move(bb));
        }
    return w.save(path, err);
}

bool GgufModel::lora_load(const std::string& path, std::string* err) {
    GgufFile f;
    if (!f.open(path, err)) return false;
    auto fail = [&](const std::string& m) {
        if (err) *err = m;
        return false;
    };
    if (f.str("adapter.type") != "lora") return fail("not a LoRA adapter");
    if (f.str("general.architecture") != cfg.arch) return fail("the adapter is for a '" + f.str("general.architecture") + "' model");
    lora_clear();
    int rank = 0;
    bool mlp = false;
    for (size_t li = 0; li < d->L.size(); ++li)
        for (int k = 0; k < 7; ++k) {
            std::string base = "blk." + std::to_string(li) + "." + kLoraNames[k] + ".weight";
            const GgufTensor* ta = f.tensor(base + ".lora_a");
            const GgufTensor* tb = f.tensor(base + ".lora_b");
            if (!ta || !tb) continue;
            int in = d->L[li].w[k].cols, out = d->L[li].w[k].rows;
            if (ta->dims.size() != 2 || tb->dims.size() != 2 || ta->dims[0] != in || tb->dims[1] != out ||
                ta->dims[1] != tb->dims[0] || !ta->data || !tb->data) {
                lora_clear();
                return fail("the adapter does not fit this model (" + base + ")");
            }
            int r = (int)ta->dims[1];
            rank = r;
            if (k >= 4) mlp = true;
            std::vector<float> at((size_t)r * in), bt((size_t)out * r);
            ggml_dequantize(ta->type, ta->data, (int64_t)at.size(), at.data());
            ggml_dequantize(tb->type, tb->data, (int64_t)bt.size(), bt.data());
            auto lo = std::make_unique<Lora>();
            lo->A.init(base + ".lora_a", in, r, false);
            lo->B.init(base + ".lora_b", r, out, false);
            for (int i = 0; i < in; ++i)
                for (int j = 0; j < r; ++j) lo->A.w.v[(size_t)i * r + j] = at[(size_t)j * in + i];
            for (int i = 0; i < r; ++i)
                for (int j = 0; j < out; ++j) lo->B.w.v[(size_t)i * out + j] = bt[(size_t)j * r + i];
            d->L[li].lo[k] = std::move(lo);
        }
    if (!rank) return fail("the adapter has no LoRA weights");
    lora_cfg.rank = rank;
    lora_cfg.mlp = mlp;
    lora_cfg.alpha = (float)f.num("adapter.lora.alpha", rank);
    d->lscale = lora_cfg.alpha / rank;
    d->cached.clear();
    return true;
}

// ------------------------------------------------------------ fine-tuning data and loop

LoraData lora_data(const GgufModel& m, const std::string& text, const std::string& q_tag, const std::string& a_tag,
                   const std::string& system) {
    LoraData d;
    auto starts = [](const std::string& s, const std::string& p) { return !p.empty() && s.compare(0, p.size(), p) == 0; };
    std::vector<std::pair<std::string, std::string>> pairs;
    if (!trim(q_tag).empty() && !trim(a_tag).empty()) {
        std::string q, a;
        int state = 0;  // 1 in a question, 2 in an answer
        auto flush = [&] {
            if (state == 2 && !trim(q).empty() && !trim(a).empty()) pairs.push_back({trim(q), trim(a)});
        };
        size_t i = 0;
        while (i <= text.size()) {
            size_t e = text.find('\n', i);
            if (e == std::string::npos) e = text.size();
            std::string line = text.substr(i, e - i);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::string lt = line;
            lt.erase(0, std::min(lt.find_first_not_of(" \t"), lt.size()));
            if (starts(lt, q_tag) || starts(lt, trim(q_tag))) {
                flush();
                q = lt.substr(starts(lt, q_tag) ? q_tag.size() : trim(q_tag).size());
                a.clear();
                state = 1;
            } else if (state >= 1 && (starts(lt, a_tag) || starts(lt, trim(a_tag)))) {
                if (state == 2) a += "\n";
                a += lt.substr(starts(lt, a_tag) ? a_tag.size() : trim(a_tag).size());
                state = 2;
            } else if (state == 1) {
                q += "\n" + line;
            } else if (state == 2) {
                a += "\n" + line;
            }
            i = e + 1;
        }
        flush();
    }
    if (!pairs.empty()) {
        for (auto& [q, a] : pairs) {
            std::vector<LlmMsg> msgs;
            if (!system.empty()) msgs.push_back({"system", system});
            msgs.push_back({"user", q});
            std::vector<int> head = m.chat_ids(msgs, true);
            msgs.push_back({"assistant", a});
            std::vector<int> full = m.chat_ids(msgs, false);
            size_t same = 0;
            while (same < head.size() && same < full.size() && head[same] == full[same]) ++same;
            if (full.size() < 2 || same >= full.size()) continue;
            if (!m.is_end(full.back()) && m.eos() >= 0) {
                // formats whose turns end in a newline: make sure the end token is learned too
                bool has_end = false;
                for (size_t k = same; k < full.size(); ++k) has_end |= m.is_end(full[k]);
                if (!has_end) full.push_back(m.eos());
            }
            std::vector<float> w(full.size(), 0.f);
            for (size_t k = same; k < full.size(); ++k) w[k] = 1.f;
            d.ex.push_back(std::move(full));
            d.w.push_back(std::move(w));
        }
        d.chat = true;
        return d;
    }
    std::vector<int> ids = m.encode(text, true, false);
    if (ids.size() >= 2) {
        d.w.push_back(std::vector<float>(ids.size(), 1.f));
        d.w.back()[0] = 0.f;
        d.ex.push_back(std::move(ids));
    }
    return d;
}

bool lora_train(GgufModel& m, const LoraData& data, const LoraConfig& cfg,
                const std::function<bool(const LoraProgress&)>& on_step, std::string* err) {
    auto fail = [&](const std::string& s) {
        if (err) *err = s;
        return false;
    };
    if (data.ex.empty() || data.tokens() < 8) return fail("not enough training text");
    LoraConfig c = cfg;
    c.batch = std::max(1, c.batch);
    c.seq = std::max(8, std::min(c.seq, m.ctx));
    // split off a validation part
    std::vector<std::vector<int>> tr_ex, va_ex;
    std::vector<std::vector<float>> tr_w, va_w;
    Rng rng(c.seed);
    if (data.chat) {
        std::vector<size_t> order(data.ex.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        for (size_t i = order.size(); i > 1; --i) std::swap(order[i - 1], order[rng.below((int)i)]);
        size_t nv = order.size() >= 10 ? order.size() / 10 : 0;
        for (size_t i = 0; i < order.size(); ++i) {
            (i < nv ? va_ex : tr_ex).push_back(data.ex[order[i]]);
            (i < nv ? va_w : tr_w).push_back(data.w[order[i]]);
        }
    } else {
        const std::vector<int>& all = data.ex[0];
        size_t nv = all.size() >= (size_t)(20 * (c.seq + 1)) ? all.size() / 10 : 0;
        tr_ex.push_back(std::vector<int>(all.begin(), all.end() - nv));
        tr_w.push_back(std::vector<float>(data.w[0].begin(), data.w[0].end() - nv));
        if (nv) {
            va_ex.push_back(std::vector<int>(all.end() - nv, all.end()));
            va_w.push_back(std::vector<float>(nv, 1.f));
            va_w.back()[0] = 0.f;
        }
    }
    int T = c.seq;
    if (data.chat) {
        size_t longest = 0;
        for (auto& e : tr_ex) longest = std::max(longest, e.size());
        T = std::max(2, std::min(T, (int)longest - 1));
    } else {
        T = std::max(2, std::min(T, (int)tr_ex[0].size() - 1));
    }
    int pad = m.eos() >= 0 ? m.eos() : 0;
    auto fill = [&](const std::vector<int>& ex, const std::vector<float>& wt, int start, int* ids, int* tg, float* w) {
        for (int t = 0; t < T; ++t) {
            int a = start + t, b = a + 1;
            ids[t] = a < (int)ex.size() ? ex[a] : pad;
            tg[t] = b < (int)ex.size() ? ex[b] : pad;
            w[t] = b < (int)ex.size() ? wt[b] : 0.f;
        }
    };
    auto start_of = [&](const std::vector<int>& ex, Rng& r) {
        int room = (int)ex.size() - (T + 1);
        return room > 0 ? r.below(room + 1) : 0;
    };
    // a fixed validation batch
    int vb = 0;
    std::vector<int> vids, vtg;
    std::vector<float> vw;
    if (!va_ex.empty()) {
        Rng vr(c.seed + 99);
        vb = std::min(8, std::max(1, (int)va_ex.size() * (data.chat ? 1 : 4)));
        vids.resize((size_t)vb * T);
        vtg.resize((size_t)vb * T);
        vw.resize((size_t)vb * T);
        for (int b = 0; b < vb; ++b) {
            const auto& ex = va_ex[b % va_ex.size()];
            fill(ex, va_w[b % va_ex.size()], start_of(ex, vr), vids.data() + (size_t)b * T, vtg.data() + (size_t)b * T, vw.data() + (size_t)b * T);
        }
    }
    if (!m.has_lora() || m.lora_cfg.rank != c.rank || m.lora_cfg.mlp != c.mlp) m.lora_init(c);
    else m.lora_cfg = c;
    std::vector<Param*> ps = m.lora_params();
    Optimizer opt(OPT_ADAMW, c.weight_decay);
    int B = c.batch;
    std::vector<int> ids((size_t)B * T), tg((size_t)B * T);
    std::vector<float> w((size_t)B * T);
    LoraProgress pr;
    pr.steps = c.steps;
    for (int step = 1; step <= c.steps; ++step) {
        auto t0 = std::chrono::steady_clock::now();
        for (int b = 0; b < B; ++b) {
            int e = rng.below((int)tr_ex.size());
            fill(tr_ex[e], tr_w[e], start_of(tr_ex[e], rng), ids.data() + (size_t)b * T, tg.data() + (size_t)b * T, w.data() + (size_t)b * T);
        }
        for (Param* p : ps) p->g.zero();
        float loss = m.lora_loss(ids.data(), tg.data(), w.data(), B, T, true);
        if (!std::isfinite(loss)) return fail("the loss became NaN");
        clip_gradients(ps, 1.f);
        int warm = std::max(1, std::min(20, c.steps / 10));
        float lr = step <= warm ? c.lr * step / warm
                                : c.lr * (0.1f + 0.45f * (1.f + std::cos(3.14159265f * (step - warm) / std::max(1, c.steps - warm))));
        opt.step(ps, lr);
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        pr.step = step;
        pr.loss = loss;
        pr.tok_s = (float)(B * T / std::max(1e-6, sec));
        if (vb && (step % std::max(1, c.eval_every) == 0 || step == c.steps))
            pr.val = m.lora_loss(vids.data(), vtg.data(), vw.data(), vb, T, false);
        if (on_step && !on_step(pr)) break;
    }
    return true;
}

}  // namespace kor
