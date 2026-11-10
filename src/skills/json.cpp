#include "json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace kor::json {

static const Value& null_value() {
    static const Value v;
    return v;
}

const Value& Value::operator[](const char* key) const {
    if (type == OBJ)
        for (auto& kv : o)
            if (kv.first == key) return kv.second;
    return null_value();
}

bool Value::has(const char* key) const {
    if (type == OBJ)
        for (auto& kv : o)
            if (kv.first == key) return true;
    return false;
}

// ---------------------------------------------------------------- reader

namespace {

struct Reader {
    const char* p;
    const char* end;
    std::string err;
    int depth = 0;

    void ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
    }
    bool fail(const char* msg) {
        if (err.empty()) err = msg;
        return false;
    }
    bool literal(const char* word) {
        size_t n = std::strlen(word);
        if ((size_t)(end - p) < n || std::strncmp(p, word, n) != 0) return fail("unexpected character");
        p += n;
        return true;
    }
    static void put_utf8(std::string& s, unsigned cp) {
        if (cp < 0x80) s += (char)cp;
        else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
        else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
    }
    bool hex4(unsigned& v) {
        if (end - p < 4) return fail("bad \\u escape");
        v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = *p++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
            else return fail("bad \\u escape");
        }
        return true;
    }
    bool string(std::string& s) {
        ++p;  // opening quote
        while (p < end && *p != '"') {
            char c = *p++;
            if (c != '\\') { s += c; continue; }
            if (p >= end) return fail("unterminated string");
            char e = *p++;
            switch (e) {
                case '"': s += '"'; break;
                case '\\': s += '\\'; break;
                case '/': s += '/'; break;
                case 'b': s += '\b'; break;
                case 'f': s += '\f'; break;
                case 'n': s += '\n'; break;
                case 'r': s += '\r'; break;
                case 't': s += '\t'; break;
                case 'u': {
                    unsigned cp;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp < 0xDC00 && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                        p += 2;
                        unsigned lo;
                        if (!hex4(lo)) return false;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    put_utf8(s, cp);
                    break;
                }
                default: return fail("bad escape");
            }
        }
        if (p >= end) return fail("unterminated string");
        ++p;
        return true;
    }
    bool value(Value& v) {
        if (++depth > 64) return fail("nested too deeply");
        ws();
        if (p >= end) return fail("unexpected end");
        bool ok = true;
        char c = *p;
        if (c == '{') {
            v.type = Value::OBJ;
            ++p;
            ws();
            if (p < end && *p == '}') { ++p; --depth; return true; }
            for (;;) {
                ws();
                if (p >= end || *p != '"') return fail("expected a key");
                std::string k;
                if (!string(k)) return false;
                ws();
                if (p >= end || *p != ':') return fail("expected ':'");
                ++p;
                v.o.emplace_back(std::move(k), Value());
                if (!value(v.o.back().second)) return false;
                ws();
                if (p < end && *p == ',') { ++p; continue; }
                if (p < end && *p == '}') { ++p; break; }
                return fail("expected ',' or '}'");
            }
        } else if (c == '[') {
            v.type = Value::ARR;
            ++p;
            ws();
            if (p < end && *p == ']') { ++p; --depth; return true; }
            for (;;) {
                v.a.emplace_back();
                if (!value(v.a.back())) return false;
                ws();
                if (p < end && *p == ',') { ++p; continue; }
                if (p < end && *p == ']') { ++p; break; }
                return fail("expected ',' or ']'");
            }
        } else if (c == '"') {
            v.type = Value::STR;
            ok = string(v.s);
        } else if (c == 't') {
            v.type = Value::BOOL; v.b = true; ok = literal("true");
        } else if (c == 'f') {
            v.type = Value::BOOL; v.b = false; ok = literal("false");
        } else if (c == 'n') {
            v.type = Value::NUL; ok = literal("null");
        } else {
            std::string num;
            while (p < end && (std::strchr("+-0123456789.eE", *p))) num += *p++;
            if (num.empty()) return fail("unexpected character");
            char* stop = nullptr;
            v.type = Value::NUM;
            v.n = std::strtod(num.c_str(), &stop);
            if (!stop || *stop) return fail("bad number");
        }
        --depth;
        return ok;
    }
};

}  // namespace

bool parse(const std::string& text, Value& out, std::string* err) {
    Reader r{text.data(), text.data() + text.size(), {}};
    out = Value();
    bool ok = r.value(out);
    if (ok) {
        r.ws();
        if (r.p != r.end) ok = r.fail("trailing characters");
    }
    if (!ok && err) *err = r.err;
    return ok;
}

// ---------------------------------------------------------------- writer

std::string quote(const std::string& s) {
    std::string q = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': q += "\\\""; break;
            case '\\': q += "\\\\"; break;
            case '\n': q += "\\n"; break;
            case '\r': q += "\\r"; break;
            case '\t': q += "\\t"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof b, "\\u%04x", c);
                    q += b;
                } else {
                    q += (char)c;
                }
        }
    }
    return q + "\"";
}

void Writer::sep() {
    if (after_key_) {
        after_key_ = false;
        return;
    }
    if (!first_.empty()) {
        if (!first_.back()) out_ += ',';
        first_.back() = false;
    }
}

Writer& Writer::begin_object() { sep(); out_ += '{'; first_.push_back(true); return *this; }
Writer& Writer::end_object() { out_ += '}'; first_.pop_back(); return *this; }
Writer& Writer::begin_array() { sep(); out_ += '['; first_.push_back(true); return *this; }
Writer& Writer::end_array() { out_ += ']'; first_.pop_back(); return *this; }

Writer& Writer::key(const char* k) {
    sep();
    out_ += quote(k);
    out_ += ':';
    after_key_ = true;
    return *this;
}

Writer& Writer::value(const std::string& v) { sep(); out_ += quote(v); return *this; }
Writer& Writer::value(const char* v) { return value(std::string(v ? v : "")); }
Writer& Writer::value(bool v) { sep(); out_ += v ? "true" : "false"; return *this; }
Writer& Writer::null() { sep(); out_ += "null"; return *this; }

Writer& Writer::value(double v) {
    sep();
    if (!std::isfinite(v)) {
        out_ += "null";
        return *this;
    }
    char b[32];
    if (v == std::floor(v) && std::fabs(v) < 1e15) std::snprintf(b, sizeof b, "%.0f", v);
    else std::snprintf(b, sizeof b, "%.9g", v);
    out_ += b;
    return *this;
}

}  // namespace kor::json
