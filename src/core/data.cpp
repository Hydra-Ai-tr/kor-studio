#include "data.h"

#include "media.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <new>
#include <set>
#include <sstream>
#include <string_view>
#include <unordered_map>

namespace kor {

static bool read_file(const std::string& path, std::string& out, std::string* err) {
    std::ifstream in(upath(path), std::ios::binary);
    if (!in) { if (err) *err = "cannot open file"; return false; }
    // straight into the string: large files are not held twice
    in.seekg(0, std::ios::end);
    std::streamoff size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size < 0) { if (err) *err = "cannot read file"; return false; }
    try {
        out.assign((size_t)size, '\0');
    } catch (const std::bad_alloc&) {
        if (err) *err = "the file does not fit in memory";
        return false;
    }
    if (size > 0 && !in.read(&out[0], size)) { if (err) *err = "cannot read file"; return false; }
    if (out.size() >= 3 && (unsigned char)out[0] == 0xEF && (unsigned char)out[1] == 0xBB && (unsigned char)out[2] == 0xBF)
        out.erase(0, 3);  // UTF-8 BOM
    return true;
}

// ------------------------------------------------------------------ text

// A character (1-4 bytes) as a number: distinct characters get distinct keys.
static uint64_t char_key(const char* s, size_t len) {
    uint64_t k = len;
    for (size_t j = 0; j < len; ++j) k = k << 8 | (unsigned char)s[j];
    return k;
}

// Calls f(pointer, length) for each character of s, '\r' left out (as utf8_chars).
template <class F>
static void each_char(const char* s, size_t n, F f) {
    for (size_t i = 0; i < n;) {
        size_t len = utf8_len(s, i, n);
        if (!(len == 1 && s[i] == '\r')) f(s + i, len);
        i += len;
    }
}

// The distinct characters of s, sorted, and (count) how many characters it has.
static std::vector<std::string> distinct_chars(const std::string& s, size_t* count) {
    std::unordered_map<uint64_t, int> seen;
    std::vector<std::string> out;
    size_t n = 0;
    each_char(s.data(), s.size(), [&](const char* c, size_t len) {
        ++n;
        if (seen.emplace(char_key(c, len), 0).second) out.emplace_back(c, len);
    });
    std::sort(out.begin(), out.end());
    if (count) *count = n;
    return out;
}

bool Dataset::load_text(const std::string& p, std::string* err, int subword) {
    std::string text;
    if (!read_file(p, text, err)) return false;
    size_t n_chars = 0;
    std::vector<std::string> uniq = distinct_chars(text, &n_chars);
    if (n_chars < 64) { if (err) *err = "text is too short (need at least 64 characters)"; return false; }
    *this = Dataset();
    kind = DATA_TEXT;
    path = p;
    raw = std::move(text);
    try {
        if (subword > (int)uniq.size()) {
            vocab = learn_subwords(raw, subword);
            ids = tokenize(raw, vocab);
        } else {
            vocab = std::move(uniq);
            std::unordered_map<uint64_t, int> map;
            for (size_t i = 0; i < vocab.size(); ++i) map[char_key(vocab[i].data(), vocab[i].size())] = (int)i;
            ids.reserve(n_chars);
            each_char(raw.data(), raw.size(), [&](const char* c, size_t len) { ids.push_back(map[char_key(c, len)]); });
        }
        split_text();
    } catch (const std::bad_alloc&) {
        *this = Dataset();
        if (err) *err = "the text is too large for the memory of this computer";
        return false;
    }
    return true;
}

// ---------------------------------------------------------- images, sound

static std::vector<std::string> media_files(const std::string& path, bool (*want)(const std::string&)) {
    namespace fs = std::filesystem;
    std::vector<std::string> files;
    std::error_code ec;
    fs::path root = upath(path);
    if (fs::is_regular_file(root, ec)) {
        if (want(root.u8string())) files.push_back(root.u8string());
        return files;
    }
    for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end; it.increment(ec))
        if (it->is_regular_file(ec) && want(it->path().u8string())) files.push_back(it->path().u8string());
    std::sort(files.begin(), files.end());
    return files;
}

static std::string hex_color(const float* c) {
    char b[8];
    std::snprintf(b, sizeof b, "#%02x%02x%02x", (int)std::lround(c[0]), (int)std::lround(c[1]), (int)std::lround(c[2]));
    return b;
}

bool Dataset::load_images(const std::string& p, int size, int colors, bool mirror, std::string* err,
                          const std::vector<std::string>* palette) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    std::vector<float> fixed;
    if (palette) {
        for (size_t i = 1; i < palette->size(); ++i) {
            unsigned r = 0, g = 0, b = 0;
            if (std::sscanf((*palette)[i].c_str(), "#%02x%02x%02x", &r, &g, &b) != 3) return fail("not an image palette");
            fixed.insert(fixed.end(), {(float)r, (float)g, (float)b});
        }
        if (fixed.empty()) return fail("not an image palette");
        colors = (int)fixed.size() / 3;
    }
    size = std::max(4, std::min(64, size));
    colors = std::max(2, std::min(256, colors));
    std::vector<std::string> files = media_files(p, is_image_file);
    std::vector<Image> imgs;
    int bad = 0;
    for (auto& f : files) {
        Image img;
        if (load_image(f, img, nullptr)) imgs.push_back(square_thumbnail(img, size));
        else ++bad;
    }
    if (imgs.size() < 2) return fail(files.empty() ? "no PNG, BMP, PPM or PGM images found" : "fewer than 2 images could be read");
    // A fixed shuffle, so the validation images are a random sample.
    Rng rng(42);
    for (int i = (int)imgs.size() - 1; i > 0; --i) std::swap(imgs[i], imgs[rng.below(i + 1)]);

    // Palette: k-means over a sample of the pixels.
    size_t npx = (size_t)size * size;
    std::vector<float> px;
    for (size_t k = 0; k < imgs.size() * npx; k += std::max<size_t>(1, imgs.size() * npx / 60000))
        for (int c = 0; c < 3; ++c) px.push_back(imgs[k / npx].rgb[(k % npx) * 3 + c]);
    size_t n = px.size() / 3;
    int K = (int)std::min<size_t>(colors, n);
    std::vector<float> cent((size_t)K * 3);
    for (int k = 0; k < K; ++k) {
        size_t s = (size_t)k * n / K;
        for (int c = 0; c < 3; ++c) cent[k * 3 + c] = px[s * 3 + c];
    }
    auto nearest = [&](const float* q) {
        int best = 0;
        float bd = 1e30f;
        for (int k = 0; k < K; ++k) {
            float d = 0;
            for (int c = 0; c < 3; ++c) { float e = q[c] - cent[k * 3 + c]; d += e * e; }
            if (d < bd) { bd = d; best = k; }
        }
        return best;
    };
    if (palette) {
        K = colors;
        cent = fixed;
    }
    std::vector<double> sum((size_t)K * 3);
    std::vector<int> cnt(K);
    for (int it = 0; it < (palette ? 0 : 12); ++it) {
        std::fill(sum.begin(), sum.end(), 0.0);
        std::fill(cnt.begin(), cnt.end(), 0);
        for (size_t i = 0; i < n; ++i) {
            int k = nearest(&px[i * 3]);
            ++cnt[k];
            for (int c = 0; c < 3; ++c) sum[k * 3 + c] += px[i * 3 + c];
        }
        for (int k = 0; k < K; ++k)
            if (cnt[k])
                for (int c = 0; c < 3; ++c) cent[k * 3 + c] = (float)(sum[k * 3 + c] / cnt[k]);
            else {  // an unused colour takes a random pixel
                size_t s = rng.next() % n;
                for (int c = 0; c < 3; ++c) cent[k * 3 + c] = px[s * 3 + c];
            }
    }

    *this = Dataset();
    kind = DATA_TEXT;
    media = MEDIA_IMAGE;
    path = p;
    img_size = size;
    sep = 0;
    vocab.push_back("<img>");
    if (palette) vocab = *palette;
    else
        for (int k = 0; k < K; ++k) vocab.push_back(hex_color(&cent[k * 3]));
    items = (int)imgs.size();
    skipped = bad;
    auto add = [&](const Image& img, bool flip) {
        ids.push_back(sep);
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x) {
                int sx = flip ? size - 1 - x : x;
                float q[3];
                for (int c = 0; c < 3; ++c) q[c] = img.rgb[((size_t)y * size + sx) * 3 + c];
                ids.push_back(1 + nearest(q));
            }
    };
    // Training images first (mirrored copies too), then the validation images.
    size_t n_val = std::max<size_t>(1, imgs.size() / 10);
    for (size_t i = 0; i + n_val < imgs.size(); ++i) {
        add(imgs[i], false);
        if (mirror) add(imgs[i], true);
    }
    size_t train_end = ids.size();
    for (size_t i = imgs.size() - n_val; i < imgs.size(); ++i) add(imgs[i], false);
    ids.push_back(sep);
    split_text();
    n_train = train_end + 1;
    return true;
}

bool Dataset::load_audio(const std::string& p, int sample_rate, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    sample_rate = std::max(2000, std::min(48000, sample_rate));
    std::vector<std::string> files = media_files(p, is_wav_file);
    std::vector<std::vector<float>> clips;
    int bad = 0;
    for (auto& f : files) {
        Sound s;
        if (!load_wav(f, s, nullptr)) { ++bad; continue; }
        std::vector<float> x = resample(s.samples, s.rate, sample_rate);
        float peak = 0;
        for (float v : x) peak = std::max(peak, std::fabs(v));
        if (peak > 1e-4f)
            for (float& v : x) v *= 0.9f / peak;  // even loudness
        if (x.size() > 64) clips.push_back(std::move(x));
    }
    if (clips.empty()) return fail(files.empty() ? "no WAV files found" : "no WAV file could be read");
    *this = Dataset();
    kind = DATA_TEXT;
    media = MEDIA_AUDIO;
    path = p;
    rate = sample_rate;
    sep = 0;
    vocab.push_back("<snd>");
    for (int k = 0; k < 256; ++k) {
        char b[8];
        std::snprintf(b, sizeof b, "~%03d", k);
        vocab.push_back(b);
    }
    items = (int)clips.size();
    skipped = bad;
    for (auto& c : clips) {
        ids.push_back(sep);
        for (float v : c) ids.push_back(1 + mulaw_encode(v));
    }
    ids.push_back(sep);
    split_text();
    return true;
}

static std::vector<int> encode(const std::string& text, const std::vector<std::string>& vocab) { return tokenize(text, vocab); }

void Dataset::split_text() {
    n_train = ids.size() - ids.size() / 10;
    layout = TextLayout();
    if (media != MEDIA_TEXT) {
        // Images and sounds: examples start after each separator token.
        for (size_t i = 0; i + 1 < ids.size(); ++i)
            if (ids[i] == sep) layout.starts.push_back(i);
        return;
    }
    std::vector<int> q, a;
    int newline = -1;
    if (tag_ids(q, a, newline)) layout = layout_dialogue(ids, q, a, newline);
}

std::vector<std::string> learn_subwords(const std::string& text, int vocab_size) {
    // Every character of the text is in the vocabulary.
    std::vector<std::string> vocab = distinct_chars(text, nullptr);
    std::unordered_map<uint64_t, int> index;
    for (size_t i = 0; i < vocab.size(); ++i) index[char_key(vocab[i].data(), vocab[i].size())] = (int)i;
    // Merges are learned on about 1M characters: the whole text when it is
    // small, otherwise 8 parts spread over it (-1 separates them). Pieces never
    // cross a line break or a part boundary.
    const size_t budget = 1500000, parts = 8;
    std::vector<int> seq;
    auto add = [&](size_t a, size_t b) {
        while (a < b && ((unsigned char)text[a] & 0xC0) == 0x80) ++a;  // start on a whole character
        if (!seq.empty()) seq.push_back(-1);
        each_char(text.data() + a, b - a, [&](const char* c, size_t len) { seq.push_back(index[char_key(c, len)]); });
    };
    if (text.size() <= budget) add(0, text.size());
    else
        for (size_t k = 0; k < parts; ++k) {
            size_t a = text.size() / parts * k;
            add(a, std::min(text.size(), a + budget / parts));
        }
    std::vector<char> nl(vocab.size());
    for (size_t i = 0; i < vocab.size(); ++i) nl[i] = vocab[i].find('\n') != std::string::npos;
    auto joins = [&](int a, int b) { return a >= 0 && b >= 0 && !nl[a] && !nl[b]; };
    std::unordered_map<uint64_t, int> count;
    std::vector<std::pair<int, uint64_t>> ranked;
    while ((int)vocab.size() < vocab_size) {
        count.clear();
        for (size_t i = 0; i + 1 < seq.size(); ++i)
            if (joins(seq[i], seq[i + 1])) ++count[(uint64_t)seq[i] << 32 | (uint32_t)seq[i + 1]];
        ranked.clear();
        for (auto& kv : count)
            if (kv.second >= 2) ranked.push_back({-kv.second, kv.first});
        if (ranked.empty()) break;
        std::sort(ranked.begin(), ranked.end());
        // Several merges per pass: the most frequent pairs that share no token
        // with each other, so one scan applies them all unambiguously.
        int room = vocab_size - (int)vocab.size();
        int want = std::max(1, std::min(64, room / 8));
        std::unordered_map<uint64_t, int> merge;
        std::vector<char> used(vocab.size() + want, 0);
        for (auto& r : ranked) {
            if ((int)merge.size() >= want) break;
            int a = (int)(r.second >> 32), b = (int)(r.second & 0xffffffffu);
            if (used[a] || used[b]) continue;
            used[a] = used[b] = 1;
            merge[r.second] = (int)vocab.size();
            vocab.push_back(vocab[a] + vocab[b]);
            nl.push_back(0);
        }
        size_t w = 0;
        for (size_t r = 0; r < seq.size(); ++r) {
            if (r + 1 < seq.size() && joins(seq[r], seq[r + 1])) {
                auto it = merge.find((uint64_t)seq[r] << 32 | (uint32_t)seq[r + 1]);
                if (it != merge.end()) { seq[w++] = it->second; ++r; continue; }
            }
            seq[w++] = seq[r];
        }
        seq.resize(w);
    }
    return vocab;
}

bool Dataset::tag_ids(std::vector<int>& q, std::vector<int>& a, int& newline) const {
    std::string qt, at;
    if (!dialogue_tags(qt, at)) return false;
    q = encode(qt, vocab);
    a = encode(at, vocab);
    newline = -1;
    for (size_t i = 0; i < vocab.size(); ++i)
        if (vocab[i] == "\n") newline = (int)i;
    return !q.empty() && !a.empty() && newline >= 0;
}

TextLayout layout_dialogue(const std::vector<int>& ids, const std::vector<int>& q, const std::vector<int>& a, int newline) {
    TextLayout lay;
    if (q.empty() || a.empty() || newline < 0) return lay;
    lay.answer.assign(ids.size(), 0);
    auto starts_with = [&](size_t pos, const std::vector<int>& tag) {
        return pos + tag.size() <= ids.size() && std::equal(tag.begin(), tag.end(), ids.begin() + (std::ptrdiff_t)pos);
    };
    for (size_t pos = 0; pos < ids.size();) {
        size_t end = pos;
        while (end < ids.size() && ids[end] != newline) ++end;
        if (starts_with(pos, q)) {
            lay.starts.push_back(pos);
        } else if (starts_with(pos, a)) {
            for (size_t i = pos + a.size(); i <= end && i < ids.size(); ++i) lay.answer[i] = 1;
        }
        pos = end + 1;
    }
    if (lay.starts.empty()) lay.answer.clear();
    return lay;
}

size_t pick_window(const TextLayout& lay, size_t lo, size_t hi, int T, float align, Rng& rng) {
    size_t last = hi - (size_t)T - 1;  // the caller guarantees hi - lo >= T + 1
    if (align > 0 && !lay.starts.empty() && rng.uniform() < align) {
        auto a = std::lower_bound(lay.starts.begin(), lay.starts.end(), lo);
        auto b = std::upper_bound(lay.starts.begin(), lay.starts.end(), last);
        if (a < b) return *(a + (std::ptrdiff_t)(rng.next() % (uint64_t)(b - a)));
    }
    return lo + (size_t)(rng.next() % (last - lo + 1));
}

void fill_window(Batch& b, int row, const std::vector<int>& ids, size_t s, const TextLayout& lay, float prompt_weight) {
    int T = b.T;
    size_t o = (size_t)row * T;
    for (int t = 0; t < T; ++t) {
        b.ids[o + t] = ids[s + t];
        b.tgt[o + t] = ids[s + t + 1];
    }
    if (b.w.empty()) return;
    bool known = lay.answer.size() == ids.size();
    for (int t = 0; t < T; ++t) b.w[o + t] = !known || lay.answer[s + t + 1] ? 1.f : prompt_weight;
}

void Dataset::encode_with(const std::vector<std::string>& fixed) {
    vocab = fixed;
    ids = encode(raw, vocab);
    split_text();
}

// -------------------------------------------------------------- dialogue

// Calls f(line) for the lines of s (as std::getline, without a trailing '\r')
// that start in its first `limit` bytes.
template <class F>
static void each_line(const std::string& s, size_t limit, F f) {
    std::string_view all(s);
    for (size_t pos = 0; pos < s.size() && pos < limit;) {
        size_t end = all.find('\n', pos);
        if (end == std::string_view::npos) end = s.size();
        std::string_view l = all.substr(pos, end - pos);
        if (!l.empty() && l.back() == '\r') l.remove_suffix(1);
        f(l);
        pos = end + 1;
    }
}

static bool starts(std::string_view l, const std::string& tag) { return l.compare(0, tag.size(), tag) == 0; }

static std::string line_tag(std::string_view l) {
    size_t p = l.find(": ");
    return p == std::string_view::npos || p == 0 || p > 24 ? "" : std::string(l.substr(0, p + 2));
}

// Dialogue structure and test questions come from the start of large texts.
static const size_t kTagScan = 8u << 20, kPairScan = 64u << 20;

bool Dataset::dialogue_tags(std::string& q_tag, std::string& a_tag) const {
    if (kind != DATA_TEXT) return false;
    std::map<std::pair<std::string, std::string>, int> count;
    std::string prev;
    size_t lines = 0;
    each_line(raw, kTagScan, [&](std::string_view l) {
        std::string b = line_tag(l);
        if (lines++ > 0 && !prev.empty() && !b.empty() && prev != b) ++count[{prev, b}];
        prev = std::move(b);
    });
    int best = 0, second = 0;
    for (auto& kv : count) {
        if (kv.second > best) { second = best; best = kv.second; q_tag = kv.first.first; a_tag = kv.first.second; }
        else if (kv.second > second) second = kv.second;
    }
    // "Q then A" must be clearly more common than "A then Q" would be by chance.
    return best >= 8 && best >= (int)(lines / 8);
}

std::vector<std::pair<std::string, std::string>> Dataset::dialogue_pairs(int n) const {
    std::vector<std::pair<std::string, std::string>> out;
    std::string qt, at;
    if (!dialogue_tags(qt, at)) return out;
    std::map<std::string, std::string> pairs;  // question -> expected ("" = ambiguous)
    std::string_view prev;
    bool have_prev = false;
    each_line(raw, kPairScan, [&](std::string_view l) {
        if (have_prev && starts(prev, qt) && starts(l, at)) {
            std::string question(prev.substr(qt.size())), expected(l.substr(at.size()));
            auto it = pairs.find(question);
            if (it == pairs.end()) pairs[question] = expected;
            else if (it->second != expected) it->second.clear();  // same question, different answers
        }
        prev = l;
        have_prev = true;
    });
    for (auto& kv : pairs)
        if (!kv.second.empty()) out.push_back(kv);
    Rng rng(123);
    for (int i = (int)out.size() - 1; i > 0; --i) std::swap(out[i], out[rng.below(i + 1)]);
    if ((int)out.size() > n) out.resize(n);
    return out;
}

std::vector<Probe> Dataset::auto_probes(int n) const {
    std::vector<Probe> out;
    std::string qt, at;
    if (!dialogue_tags(qt, at)) return out;
    for (auto& qa : dialogue_pairs(n)) out.push_back(make_probe(qa.first, qa.second, qt, at));
    return out;
}

static std::string trim(std::string_view s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string_view::npos ? "" : std::string(s.substr(a, b - a + 1));
}

int Dataset::hold_out(const std::vector<Probe>& tests) {
    if (kind != DATA_TEXT || tests.empty()) return 0;
    std::string qt, at;
    bool dialogue = dialogue_tags(qt, at);
    std::set<std::string> questions;
    for (auto& p : tests)
        if (!p.question.empty()) questions.insert(p.question);
    auto asks = [&](std::string_view line) {
        if (dialogue) return starts(line, qt) && questions.count(trim(line.substr(qt.size()))) > 0;
        std::string l = trim(line);
        for (auto& q : questions)
            if (l.compare(0, q.size(), q) == 0) return true;
        return false;
    };
    // The kept lines (each followed by '\n'); with `keep` null only counts the removed ones.
    auto scan = [&](std::string* keep) {
        int removed = 0;
        bool skip_answer = false;
        each_line(raw, raw.size(), [&](std::string_view l) {
            if (skip_answer) {
                skip_answer = false;
                if (starts(l, at)) return;
            }
            if (asks(l)) {
                ++removed;
                skip_answer = dialogue;
                return;
            }
            if (keep) {
                keep->append(l.data(), l.size());
                *keep += '\n';
            }
        });
        return removed;
    };
    int removed = scan(nullptr);
    if (removed) {
        std::string kept;
        kept.reserve(raw.size());
        scan(&kept);
        ids = encode(kept, vocab);
        split_text();
    }
    return removed;
}

Probe make_probe(const std::string& question, const std::string& expected, const std::string& q_tag, const std::string& a_tag) {
    Probe p;
    p.question = trim(question);
    p.expected = trim(expected);
    p.prompt = q_tag.empty() && a_tag.empty() ? p.question : q_tag + p.question + "\n" + a_tag;
    return p;
}

std::vector<Probe> parse_tests(const std::string& text, const std::string& q_tag, const std::string& a_tag) {
    std::vector<Probe> out;
    each_line(text, text.size(), [&](std::string_view line) {
        std::string l = trim(line);
        if (l.empty() || l[0] == '#') return;
        size_t sep = l.find("=>");
        if (sep == std::string::npos) return;
        Probe p = make_probe(l.substr(0, sep), l.substr(sep + 2), q_tag, a_tag);
        if (!p.question.empty() && !p.expected.empty()) out.push_back(p);
    });
    return out;
}

std::string format_tests(const std::vector<std::pair<std::string, std::string>>& pairs) {
    std::string s;
    for (auto& qa : pairs) s += qa.first + " => " + qa.second + "\n";
    return s;
}

// ----------------------------------------------------------------- table

static std::vector<std::string> split_csv(const std::string& line, char delim) {
    std::vector<std::string> out;
    std::string cur;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (quoted) {
            if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') { cur += '"'; ++i; }
            else if (c == '"') quoted = false;
            else cur += c;
        } else if (c == '"') quoted = true;
        else if (c == delim) { out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    out.push_back(cur);
    for (auto& s : out) {
        size_t a = s.find_first_not_of(" \t"), b = s.find_last_not_of(" \t");
        s = a == std::string::npos ? "" : s.substr(a, b - a + 1);
    }
    return out;
}

static bool parse_num(const std::string& s, float& v) {
    if (s.empty()) return false;
    char* end = nullptr;
    double d = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0' || !std::isfinite(d)) return false;
    v = (float)d;
    return true;
}

bool Dataset::load_table(const std::string& p, int tcol, std::string* err) {
    std::string text;
    if (!read_file(p, text, err)) return false;
    std::vector<std::string> lines;
    {
        std::istringstream in(text);
        std::string l;
        while (std::getline(in, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (!l.empty()) lines.push_back(l);
        }
    }
    if (lines.size() < 4) { if (err) *err = "table needs at least 4 rows"; return false; }
    char delim = ',';
    size_t best = 0;
    for (char d : {',', ';', '\t'}) {
        size_t c = (size_t)std::count(lines[0].begin(), lines[0].end(), d);
        if (c > best) { best = c; delim = d; }
    }
    std::vector<std::vector<std::string>> rows;
    for (auto& l : lines) rows.push_back(split_csv(l, delim));
    int nc = (int)rows[0].size();
    if (nc < 2) { if (err) *err = "table needs at least 2 columns"; return false; }

    Dataset d;
    d.kind = DATA_TABLE;
    d.path = p;
    d.n_cols = nc;
    bool has_header = false;
    float tmp;
    for (auto& c : rows[0]) if (!parse_num(c, tmp)) has_header = true;
    if (has_header) { d.header = rows[0]; rows.erase(rows.begin()); }
    else for (int i = 0; i < nc; ++i) d.header.push_back("col" + std::to_string(i + 1));
    rows.erase(std::remove_if(rows.begin(), rows.end(), [&](const std::vector<std::string>& r) { return (int)r.size() != nc; }), rows.end());
    int n = (int)rows.size();
    if (n < 4) { if (err) *err = "not enough valid rows"; return false; }
    for (int i = 0; i < std::min(n, 8); ++i) d.preview.push_back(rows[i]);

    if (tcol < 0 || tcol >= nc) tcol = nc - 1;
    d.target_col = tcol;
    d.n_feat = nc - 1;

    // Decide the task: numeric target with many distinct values => regression.
    bool numeric = true;
    std::set<std::string> distinct;
    for (auto& r : rows) {
        if (!parse_num(r[tcol], tmp)) numeric = false;
        if (distinct.size() <= 64) distinct.insert(r[tcol]);
    }
    d.regression = numeric && (int)distinct.size() > std::min(20, std::max(2, n / 4));

    d.X.resize(n, d.n_feat);
    for (int i = 0; i < n; ++i) {
        int j = 0;
        for (int c = 0; c < nc; ++c) {
            if (c == tcol) continue;
            float v = 0;
            if (!parse_num(rows[i][c], v)) { v = 0; ++d.bad_cells; }
            d.X.row(i)[j++] = v;
        }
    }
    d.mean.assign(d.n_feat + 1, 0.f);
    d.stdv.assign(d.n_feat + 1, 1.f);
    for (int j = 0; j < d.n_feat; ++j) {
        double m = 0, s = 0;
        for (int i = 0; i < n; ++i) m += d.X.row(i)[j];
        m /= n;
        for (int i = 0; i < n; ++i) { double e = d.X.row(i)[j] - m; s += e * e; }
        s = std::sqrt(s / n);
        if (s < 1e-8) s = 1;
        d.mean[j] = (float)m; d.stdv[j] = (float)s;
        for (int i = 0; i < n; ++i) d.X.row(i)[j] = (float)((d.X.row(i)[j] - m) / s);
    }
    if (d.regression) {
        d.yv.resize(n);
        double m = 0, s = 0;
        for (int i = 0; i < n; ++i) { parse_num(rows[i][tcol], d.yv[i]); m += d.yv[i]; }
        m /= n;
        for (int i = 0; i < n; ++i) { double e = d.yv[i] - m; s += e * e; }
        s = std::sqrt(s / n);
        if (s < 1e-8) s = 1;
        d.mean[d.n_feat] = (float)m; d.stdv[d.n_feat] = (float)s;
        for (auto& y : d.yv) y = (float)((y - m) / s);
    } else {
        std::map<std::string, int> map;
        for (auto& r : rows) map.emplace(r[tcol], 0);
        if (map.size() < 2) { if (err) *err = "target column has only one class"; return false; }
        int k = 0;
        for (auto& kv : map) { kv.second = k++; d.labels.push_back(kv.first); }
        d.cls.resize(n);
        for (int i = 0; i < n; ++i) d.cls[i] = map[rows[i][tcol]];
    }
    // Deterministic shuffled 85/15 split.
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    Rng rng(42);
    for (int i = n - 1; i > 0; --i) std::swap(idx[i], idx[rng.below(i + 1)]);
    int nv = std::max(1, n * 15 / 100);
    d.val_idx.assign(idx.begin(), idx.begin() + nv);
    d.train_idx.assign(idx.begin() + nv, idx.end());
    *this = std::move(d);
    return true;
}

// -------------------------------------------------------------- sampling

size_t Dataset::train_items() const {
    return kind == DATA_TEXT ? n_train : kind == DATA_TABLE ? train_idx.size() : 0;
}
size_t Dataset::val_items() const {
    return kind == DATA_TEXT ? ids.size() - n_train : kind == DATA_TABLE ? val_idx.size() : 0;
}

void Dataset::sample(Batch& b, int B, int T, Rng& rng, bool val, const SampleOpts& opts) const {
    if (kind == DATA_TEXT) {
        // Image training ends one token into the validation part, on the
        // separator that starts the first validation image.
        size_t lo = val ? n_train - (media == MEDIA_IMAGE ? 1 : 0) : 0, hi = val ? ids.size() : n_train;
        if (hi - lo < (size_t)T + 1) { lo = 0; hi = ids.size(); }  // tiny corpus: reuse everything
        b.B = B; b.T = T;
        b.ids.resize((size_t)B * T);
        b.tgt.resize((size_t)B * T);
        bool weighted = opts.prompt_weight != 1.f && !layout.empty();
        if (weighted) b.w.resize((size_t)B * T);
        else b.w.clear();
        for (int i = 0; i < B; ++i) fill_window(b, i, ids, pick_window(layout, lo, hi, T, opts.align, rng), layout, opts.prompt_weight);
    } else if (kind == DATA_TABLE) {
        b.w.clear();
        const std::vector<int>& idx = val ? val_idx : train_idx;
        b.B = B; b.T = 1;
        b.x.resize_noinit(B, n_feat);
        if (regression) b.y.resize_noinit(B, 1);
        else b.tgt.resize(B);
        for (int i = 0; i < B; ++i) {
            int r = idx[rng.below((int)idx.size())];
            std::copy(X.row(r), X.row(r) + n_feat, b.x.row(i));
            if (regression) b.y.v[i] = yv[r];
            else b.tgt[i] = cls[r];
        }
    }
}

bool write_spiral_csv(const std::string& path, int per_class) {
    std::ofstream o(upath(path), std::ios::binary);
    if (!o) return false;
    o << "x,y,class\n";
    Rng rng(7);
    const char* names[3] = {"red", "green", "blue"};
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < per_class; ++i) {
            float r = (float)i / per_class;
            float t = c * 2.0944f + r * 4.2f + rng.normal() * 0.18f;
            o << r * std::cos(t) << "," << r * std::sin(t) << "," << names[c] << "\n";
        }
    return (bool)o;
}

}  // namespace kor
