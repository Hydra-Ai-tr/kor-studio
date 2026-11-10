// Kor Studio - a small JSON reader and writer for the skill protocol.
#pragma once
#include <string>
#include <utility>
#include <vector>

namespace kor::json {

struct Value {
    enum Type { NUL, BOOL, NUM, STR, ARR, OBJ };
    Type type = NUL;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Value> a;
    std::vector<std::pair<std::string, Value>> o;

    // Missing keys (or a non-object) give a null value.
    const Value& operator[](const char* key) const;
    bool has(const char* key) const;
    double num(double def = 0) const { return type == NUM ? n : type == BOOL ? (b ? 1 : 0) : def; }
    const std::string& str() const { return s; }
    bool flag(bool def = false) const { return type == BOOL ? b : type == NUM ? n != 0 : def; }
};

bool parse(const std::string& text, Value& out, std::string* err = nullptr);

// Builds JSON text; commas are inserted automatically.
class Writer {
public:
    Writer& begin_object();
    Writer& end_object();
    Writer& begin_array();
    Writer& end_array();
    Writer& key(const char* k);
    Writer& value(const std::string& v);
    Writer& value(const char* v);
    Writer& value(double v);
    Writer& value(int v) { return value((double)v); }
    Writer& value(long long v) { return value((double)v); }
    Writer& value(size_t v) { return value((double)v); }
    Writer& value(bool v);
    Writer& null();
    template <class T>
    Writer& field(const char* k, const T& v) { return key(k).value(v); }
    const std::string& str() const { return out_; }

private:
    void sep();
    std::string out_;
    std::vector<bool> first_;
    bool after_key_ = false;
};

std::string quote(const std::string& s);

}  // namespace kor::json
