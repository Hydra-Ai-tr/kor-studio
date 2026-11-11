#include "datatools.h"

#include "json.h"

#include <set>
#include <vector>

namespace kor {
namespace {

std::vector<std::string> lines_of(const std::string& text) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (pos <= text.size()) {
        size_t e = text.find('\n', pos);
        if (e == std::string::npos) e = text.size();
        std::string l = text.substr(pos, e - pos);
        if (!l.empty() && l.back() == '\r') l.pop_back();
        out.push_back(std::move(l));
        pos = e + 1;
    }
    if (!out.empty() && out.back().empty()) out.pop_back();
    return out;
}

// Tabs, no-break and other odd spaces become plain spaces; control characters go away.
std::string tidy(const std::string& s, bool collapse) {
    std::string o;
    o.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = (unsigned char)s[i];
        if (c == '\t') c = ' ';
        else if (c < 32 || c == 127) continue;
        else if (c == 0xC2 && i + 1 < s.size() && (unsigned char)s[i + 1] == 0xA0) { c = ' '; ++i; }        // U+00A0
        else if (c == 0xE2 && i + 2 < s.size() && (unsigned char)s[i + 1] == 0x80) {
            unsigned char d = (unsigned char)s[i + 2];
            if (d >= 0x80 && d <= 0x8B) { c = ' '; i += 2; }                                              // U+2000..200B
            else if (d == 0xAF) { c = ' '; i += 2; }                                                        // U+202F
        } else if (c == 0xEF && i + 2 < s.size() && (unsigned char)s[i + 1] == 0xBB && (unsigned char)s[i + 2] == 0xBF) {
            i += 2;                                                                                          // BOM
            continue;
        }
        if (c == ' ' && collapse && !o.empty() && o.back() == ' ') continue;
        o += (char)c;
    }
    return o;
}

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(' '), b = s.find_last_not_of(' ');
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

bool starts(const std::string& s, const std::string& p) { return !p.empty() && s.compare(0, p.size(), p) == 0; }

}  // namespace

std::string clean_text(const std::string& text, const std::string& q_tag, const std::string& a_tag, const CleanOptions& o,
                       CleanReport& r) {
    r = CleanReport{};
    r.bytes_in = text.size();
    std::vector<std::string> in = lines_of(text);
    r.lines_in = in.size();
    std::vector<std::string> lines;
    for (const std::string& raw : in) {
        std::string l = o.trim ? trim(tidy(raw, o.collapse_spaces)) : raw;
        if (l != raw) ++r.changed;
        lines.push_back(std::move(l));
    }
    std::string out;
    bool dialogue = !q_tag.empty() && !a_tag.empty();
    if (dialogue) {
        // tags may have lost their trailing space to trimming
        std::string qt = trim(q_tag), at = trim(a_tag);
        std::set<std::string> seen;
        for (size_t i = 0; i < lines.size(); ++i) {
            const std::string& l = lines[i];
            if (l.empty()) {
                ++r.empty;
                if (!o.drop_empty) out += "\n";
                continue;
            }
            bool q = starts(l, qt), next_a = i + 1 < lines.size() && starts(lines[i + 1], at);
            if (q && next_a) {
                std::string pair = l + "\n" + lines[i + 1] + "\n";
                ++i;
                if (o.dedupe && !seen.insert(pair).second) {
                    ++r.duplicates;
                    continue;
                }
                out += pair;
                r.lines_out += 2;
                continue;
            }
            if (o.drop_broken && (q || starts(l, at))) {
                ++r.broken;
                continue;
            }
            out += l + "\n";
            ++r.lines_out;
        }
    } else {
        std::set<std::string> seen;
        bool last_empty = true;
        for (const std::string& l : lines) {
            if (l.empty()) {
                if (o.drop_empty && last_empty) {
                    ++r.empty;
                    continue;
                }
                out += "\n";
                ++r.lines_out;
                last_empty = true;
                continue;
            }
            // only lines long enough to be real sentences count as repeats
            if (o.dedupe && l.size() >= 24 && !seen.insert(l).second) {
                ++r.duplicates;
                continue;
            }
            out += l + "\n";
            ++r.lines_out;
            last_empty = false;
        }
        while (out.size() >= 2 && out[out.size() - 1] == '\n' && out[out.size() - 2] == '\n') {
            out.pop_back();
            --r.lines_out;
        }
    }
    r.bytes_out = out.size();
    return out;
}

namespace {

// Message content: a string, or a list of parts with "text".
std::string content_of(const json::Value& v) {
    if (v.type == json::Value::STR) return v.s;
    std::string s;
    if (v.type == json::Value::ARR)
        for (const json::Value& p : v.a) {
            const std::string& t = p.type == json::Value::STR ? p.s : p["text"].str();
            if (!t.empty()) s += (s.empty() ? "" : " ") + t;
        }
    return s;
}

std::string one_line(const std::string& s) {
    std::string o;
    for (char c : s) o += c == '\n' || c == '\r' || c == '\t' ? ' ' : c;
    return trim(tidy(o, true));
}

}  // namespace

bool import_jsonl(const std::string& json_text, const std::string& q_tag, const std::string& a_tag, std::string& out,
                  ImportReport& r, std::string* err) {
    r = ImportReport{};
    out.clear();
    std::vector<json::Value> records;
    size_t first = json_text.find_first_not_of(" \t\r\n\xEF\xBB\xBF");
    if (first == std::string::npos) {
        if (err) *err = "the file is empty";
        return false;
    }
    if (json_text[first] == '[') {
        json::Value all;
        std::string e;
        if (!json::parse(json_text.substr(first), all, &e) || all.type != json::Value::ARR) {
            if (err) *err = "invalid JSON: " + e;
            return false;
        }
        records = std::move(all.a);
    } else {
        for (const std::string& line : lines_of(json_text)) {
            if (line.find_first_not_of(" \t") == std::string::npos) continue;
            json::Value v;
            if (json::parse(line, v) && v.type == json::Value::OBJ) records.push_back(std::move(v));
            else ++r.skipped;
        }
    }
    std::set<std::string> formats;
    auto turn = [&](bool user, const std::string& text) {
        std::string t = one_line(text);
        if (t.empty()) return false;
        out += (user ? q_tag : a_tag) + t + "\n";
        ++r.turns;
        return true;
    };
    for (const json::Value& v : records) {
        ++r.records;
        size_t before = out.size(), turns_before = r.turns;
        if (v["messages"].type == json::Value::ARR || v["conversations"].type == json::Value::ARR) {
            bool share = v["messages"].type != json::Value::ARR;
            formats.insert(share ? "sharegpt" : "messages");
            const json::Value& list = share ? v["conversations"] : v["messages"];
            bool want_user = true;
            for (const json::Value& m : list.a) {
                std::string role = share ? m["from"].str() : m["role"].str();
                std::string text = share ? m["value"].str() : content_of(m["content"]);
                bool user = role == "user" || role == "human";
                bool bot = role == "assistant" || role == "gpt" || role == "bot" || role == "model";
                if (!user && !bot) continue;  // system prompts and tools
                if (user != want_user) continue;  // keep strict question/answer turns
                if (turn(user, text)) want_user = !want_user;
            }
            // a dialogue must end with an answer
            if (!want_user && r.turns > turns_before) {
                size_t cut = out.rfind('\n', out.size() - 2);
                out.resize(cut == std::string::npos || cut < before ? before : cut + 1);
                --r.turns;
            }
        } else if (v.has("instruction") && v.has("output")) {
            formats.insert("alpaca");
            std::string q = v["instruction"].str(), in = v["input"].str();
            if (!in.empty()) q += " " + in;
            if (!one_line(q).empty() && !one_line(v["output"].str()).empty()) {
                turn(true, q);
                turn(false, v["output"].str());
            }
        } else if ((v.has("prompt") && (v.has("completion") || v.has("response"))) || (v.has("question") && v.has("answer"))) {
            formats.insert("prompt");
            bool qa = v.has("question");
            std::string q = qa ? v["question"].str() : v["prompt"].str();
            std::string a = qa ? v["answer"].str() : v.has("completion") ? v["completion"].str() : v["response"].str();
            if (!one_line(q).empty() && !one_line(a).empty()) {
                turn(true, q);
                turn(false, a);
            }
        } else if (v["text"].type == json::Value::STR) {
            formats.insert("text");
            std::string t = v["text"].str();
            std::string clean;
            for (char c : t)
                if (c != '\r') clean += c;
            if (!trim(clean).empty()) {
                out += clean;
                out += clean.back() == '\n' ? "\n" : "\n\n";
                ++r.turns;
            }
        }
        if (out.size() == before) ++r.skipped;
        else ++r.dialogues;
    }
    r.format = formats.size() == 1 ? *formats.begin() : formats.empty() ? "" : "mixed";
    if (out.empty()) {
        if (err) *err = records.empty() ? "no JSON records found" : "no records in a known format (messages, conversations, instruction/output, prompt/completion, question/answer, text)";
        return false;
    }
    return true;
}

}  // namespace kor
