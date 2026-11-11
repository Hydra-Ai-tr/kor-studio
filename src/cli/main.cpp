// kor-cli: headless companion to Kor Studio (system info, training, self-test,
// generation and question tests). Also used by Python plugins.
#include "../core/gguf.h"
#include "../core/gpu.h"
#include "../core/kernels.h"
#include "../core/media.h"
#include "../core/parallel.h"
#include "../core/sysinfo.h"
#include "../skills/datatools.h"
#include "../skills/skill.h"
#include "serve.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#define PATH_SEP ";"
#else
#define PATH_SEP ":"
#endif

using namespace kor;

static std::atomic<bool> g_interrupt{false};
static void on_interrupt(int) { g_interrupt = true; }

// Command-line arguments as UTF-8 (on Windows argv uses the ANSI code page).
static std::vector<std::string> utf8_args(int argc, char** argv) {
    std::vector<std::string> args;
#if defined(_WIN32)
    int n = 0;
    if (wchar_t** w = CommandLineToArgvW(GetCommandLineW(), &n)) {
        for (int i = 0; i < n; ++i) {
            int len = WideCharToMultiByte(CP_UTF8, 0, w[i], -1, nullptr, 0, nullptr, nullptr);
            std::string s(len > 0 ? (size_t)len - 1 : 0, '\0');
            if (len > 1) WideCharToMultiByte(CP_UTF8, 0, w[i], -1, &s[0], len, nullptr, nullptr);
            args.push_back(s);
        }
        LocalFree(w);
        return args;
    }
#endif
    for (int i = 0; i < argc; ++i) args.push_back(argv[i]);
    return args;
}

static bool read_text(const std::string& path, std::string& out) {
    std::ifstream in(upath(path), std::ios::binary);
    if (!in) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

// The value after a flag, or "" when the flag is absent.
static std::string flag(const std::vector<std::string>& a, const char* name, const char* def = "") {
    for (size_t i = 0; i + 1 < a.size(); ++i)
        if (a[i] == name) return a[i + 1];
    return def;
}
static bool has_flag(const std::vector<std::string>& a, const char* name) {
    for (auto& s : a)
        if (s == name) return true;
    return false;
}
// Every value given after a flag that may be repeated.
static std::vector<std::string> flags(const std::vector<std::string>& a, const char* name) {
    std::vector<std::string> out;
    for (size_t i = 0; i + 1 < a.size(); ++i)
        if (a[i] == name) out.push_back(a[++i]);
    return out;
}

// Python skills given with --skill "file.py [args]".
static bool add_skills(const std::vector<std::string>& a, const std::string& data_path, const Dataset& data,
                       TrainConfig& cfg) {
    std::vector<std::string> specs = flags(a, "--skill");
    if (specs.empty()) return true;
    std::string version;
    std::vector<std::string> python = has_flag(a, "--python") ? split_command(flag(a, "--python")) : find_python3(&version);
    if (python.empty()) {
        std::printf("error: Python 3 was not found (install it or pass --python <command>)\n");
        return false;
    }
    namespace fs = std::filesystem;
    fs::path dir = upath(program_dir());
    std::string lib = (dir / "python").u8string();
    if (!fs::exists(dir / "python" / "kor")) {
        fs::path src = dir.parent_path().parent_path() / "python";  // running from build/bin in the source tree
        if (fs::exists(src / "kor")) lib = src.u8string();
    }
    std::string pypath = lib;
    if (const char* old = std::getenv("PYTHONPATH"))
        if (*old) pypath += std::string(PATH_SEP) + old;
#if defined(_WIN32)
    std::string cli = (dir / "kor-cli.exe").u8string();
#else
    std::string cli = (dir / "kor-cli").u8string();
#endif
    std::string qt, at;
    data.dialogue_tags(qt, at);
    for (auto& s : specs) {
        std::vector<std::string> parts = split_command(s);
        if (parts.empty()) continue;
        if (!fs::exists(upath(parts[0]))) {
            std::printf("error: skill not found: %s\n", parts[0].c_str());
            return false;
        }
        SkillSpec sp;
        sp.script = parts[0];
        sp.name = upath(parts[0]).stem().u8string();
        sp.args.assign(parts.begin() + 1, parts.end());
        sp.python = python;
        sp.lang = flag(a, "--lang", "en");
        sp.q_tag = qt;
        sp.a_tag = at;
        sp.env = {{"PYTHONPATH", pypath}, {"KOR_PYLIB", lib}, {"KOR_CLI", cli}, {"KOR_DATA", data_path}};
        cfg.hooks.push_back(std::make_shared<PythonSkill>(sp));
        std::string shown = sp.name;
        for (auto& x : sp.args) shown += " " + x;
        std::printf("skill: %s\n", shown.c_str());
    }
    if (!version.empty()) std::printf("python: %s\n", version.c_str());
    return true;
}

static void print_sysinfo() {
    SysInfo s = detect_system();
    std::printf("CPU      : %s\n", s.cpu.c_str());
    std::printf("Threads  : %d logical, %d recommended\n", s.logical, recommended_threads(s));
    std::printf("RAM      : %.1f GB total, %.1f GB free\n", s.ram_total_gb, s.ram_free_gb);
    std::printf("SIMD     : %s%s(kernels: %s)\n", s.avx2 ? "AVX2 " : "", s.avx512 ? "AVX-512 " : "", K().name);
    for (auto& g : s.gpus) std::printf("GPU      : %s\n", g.c_str());
}

// Compares analytic gradients against central finite differences.
static double gradcheck(Model& m, const Batch& b, const char* label) {
    m.zero_grad();
    m.step(b, true, 1.f);
    Rng rng(5);
    double worst = 0;
    for (Param* p : m.params()) {
        if (!p->trainable) continue;
        double pworst = 0;
        for (int k = 0; k < 6; ++k) {
            size_t i = (size_t)rng.below((int)p->w.size());
            float old = p->w.v[i];
            const float eps = 4e-3f;  // small enough to rarely straddle a ReLU kink
            p->w.v[i] = old + eps;
            double lp = m.step(b, false, 1.f);
            p->w.v[i] = old - eps;
            double lm = m.step(b, false, 1.f);
            p->w.v[i] = old;
            double num = (lp - lm) / (2.0 * eps), ana = p->g.v[i];
            double err = std::fabs(num - ana) / std::max(2e-3, std::fabs(num) + std::fabs(ana));
            pworst = std::max(pworst, err);
        }
        if (pworst > 0.08) std::printf("  [%s] %-16s rel.err %.4f\n", label, p->name.c_str(), pworst);
        worst = std::max(worst, pworst);
    }
    std::printf("gradcheck %-14s worst relative error %.4f %s\n", label, worst, worst < 0.08 ? "OK" : "FAIL");
    return worst;
}

// Largest weight difference between a run stopped halfway and resumed and an
// uninterrupted one (-1 when something failed). With `gpu` the weights and the
// optimizer stay on the card (no EMA) and the stopping hook also generates text.
static float resume_check(bool gpu, std::string& err) {
    std::string text;
    for (int i = 0; i < 120; ++i) text += "Kullanıcı: merhaba " + std::to_string(i % 7) + "\nAsistan: selam " + std::to_string(i % 5) + "\n";
    std::string dpath = "kor_selftest_data.txt", ck = "kor_selftest_ckpt.kor";
    { std::ofstream f(upath(dpath), std::ios::binary); f << text; }
    Dataset data;
    bool pass = data.load_text(dpath, &err);
    std::remove(dpath.c_str());
    struct StopAt : TrainHook {
        Trainer* tr = nullptr;
        int at = 0;
        std::string name() const override { return "stop-at"; }
        bool open(TrainControl&, std::string*) override { return true; }
        bool wants(int k) const override { return k == HOOK_STEP; }
        int step_every() const override { return at; }
        void on_event(const HookInfo&, TrainControl& ctl) override {
            ctl.generate("Kullanıcı:", 4, 0.f, true);
            tr->request_stop();
        }
    };
    auto run = [&](bool interrupt, std::vector<float>& w_out) {
        ModelSpec s;
        s.kind = MODEL_GPT; s.vocab = (int)data.vocab.size(); s.ctx = 16; s.dim = 16; s.heads = 2; s.nlayer = 1;
        auto m = std::make_shared<Model>(s);
        m->vocab = data.vocab;
        TrainConfig c;
        c.steps = 40; c.batch = 4; c.eval_every = 10; c.ema = gpu ? 0.f : 0.9f; c.threads = 1; c.gpu = gpu;
        Trainer tr;
        auto finish = [&] {
            while (tr.running()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            tr.stop();
        };
        if (interrupt) {
            auto h = std::make_shared<StopAt>();
            h->tr = &tr;
            h->at = 20;
            c.hooks.push_back(h);
            c.checkpoint = ck;
            if (!tr.start(m, &data, c)) return false;
            finish();
            if (!has_checkpoint(ck)) return false;
            CheckpointInfo info;
            if (!read_checkpoint(ck, info, &err) || info.step != 20) return false;
            m = Model::load(ck, &err);
            if (!m) return false;
            c = info.cfg;
            c.threads = 1;
            c.gpu = gpu;  // a setting of the machine, not saved with the run
            c.checkpoint = ck;
            c.resume = true;
        }
        if (!tr.start(m, &data, c)) return false;
        finish();
        if (tr.stats().state != TS_DONE || has_checkpoint(ck)) return false;
        for (Param* p : m->params()) w_out.insert(w_out.end(), p->w.v.begin(), p->w.v.end());
        return true;
    };
    std::vector<float> wa, wb;
    pass = pass && run(false, wa) && run(true, wb) && wa.size() == wb.size();
    float diff = 0;
    for (size_t i = 0; pass && i < wa.size(); ++i) diff = std::max(diff, std::fabs(wa[i] - wb[i]));
    remove_checkpoint(ck);
    return pass ? diff : -1.f;
}

// quick leaves out the checks that train a model for a while.
// ---- GGUF: our modern transformer written as a llama / qwen2 GGUF file must give the same results.

struct TestVocab {
    std::string model = "gpt2", pre = "qwen2", tmpl;
    std::vector<std::string> toks, merges;
    std::vector<int32_t> types;
    std::vector<float> scores;
};

static bool export_gguf_test(Model& m, const std::string& path, const std::string& arch, int type, const TestVocab& tv,
                             std::string* err) {
    const ModelSpec& s = m.spec;
    int C = s.dim, F = s.mlp_hidden(), V = s.vocab, hd = C / s.heads, hh = hd / 2;
    std::map<std::string, Param*> P;
    for (Param* p : m.params()) P[p->name] = p;
    GgufWriter w;
    w.set_str("general.architecture", arch);
    w.set_str("general.name", "kor-selftest");
    std::string a = arch + ".";
    w.set_u32(a + "context_length", s.ctx);
    w.set_u32(a + "embedding_length", C);
    w.set_u32(a + "block_count", s.nlayer);
    w.set_u32(a + "feed_forward_length", F);
    w.set_u32(a + "attention.head_count", s.heads);
    w.set_u32(a + "attention.head_count_kv", s.heads);
    w.set_f32(a + "rope.freq_base", s.rope_theta);
    w.set_f32(a + "attention.layer_norm_rms_epsilon", 1e-5f);
    w.set_str("tokenizer.ggml.model", tv.model);
    w.set_str("tokenizer.ggml.pre", tv.pre);
    w.set_strs("tokenizer.ggml.tokens", tv.toks);
    w.set_i32s("tokenizer.ggml.token_type", tv.types);
    if (!tv.scores.empty()) w.set_f32s("tokenizer.ggml.scores", tv.scores);
    if (!tv.merges.empty()) w.set_strs("tokenizer.ggml.merges", tv.merges);
    if (!tv.tmpl.empty()) w.set_str("tokenizer.chat_template", tv.tmpl);
    if (tv.model == "llama") {
        w.set_u32("tokenizer.ggml.bos_token_id", 1);
        w.set_u32("tokenizer.ggml.eos_token_id", 2);
    }
    // rows of `out` values taken from columns [c0, c0 + out) of W [in x out_all]; llama rotates (2i, 2i+1) pairs, we rotate (i, i + hd/2)
    auto rows_of = [&](const Tensor& W, int c0, int out, bool perm) {
        std::vector<float> r((size_t)out * W.rows);
        for (int o = 0; o < out; ++o) {
            int src = o;
            if (perm) {
                int head = o / hd, j = o % hd;
                src = head * hd + (j % 2 ? j / 2 + hh : j / 2);
            }
            for (int i = 0; i < W.rows; ++i) r[(size_t)o * W.rows + i] = W.v[(size_t)i * W.cols + c0 + src];
        }
        return r;
    };
    auto add = [&](const std::string& name, const std::vector<float>& v, int64_t cols, int64_t rows, int t) {
        std::vector<uint8_t> b;
        ggml_quantize(t, v.data(), (int64_t)v.size(), b);
        if (rows > 1) w.add_tensor(name, {cols, rows}, t, std::move(b));
        else w.add_tensor(name, {cols}, t, std::move(b));
    };
    bool perm = arch == "llama";
    const Tensor& tok = P["emb.tok"]->w;
    add("token_embd.weight", tok.v, C, V, type);
    for (int l = 0; l < s.nlayer; ++l) {
        std::string h = "h" + std::to_string(l) + ".", b = "blk." + std::to_string(l) + ".";
        const Tensor& qkv = P[h + "attn.qkv.W"]->w;
        const Tensor& qb = P[h + "attn.qkv.b"]->w;
        add(b + "attn_norm.weight", P[h + "ln1.g"]->w.v, C, 1, GGML_F32);
        add(b + "attn_q.weight", rows_of(qkv, 0, C, perm), C, C, type);
        add(b + "attn_k.weight", rows_of(qkv, C, C, perm), C, C, type);
        add(b + "attn_v.weight", rows_of(qkv, 2 * C, C, false), C, C, type);
        add(b + "attn_q.bias", rows_of(qb, 0, C, perm), C, 1, GGML_F32);
        add(b + "attn_k.bias", rows_of(qb, C, C, perm), C, 1, GGML_F32);
        add(b + "attn_v.bias", rows_of(qb, 2 * C, C, false), C, 1, GGML_F32);
        add(b + "attn_output.weight", rows_of(P[h + "attn.proj.W"]->w, 0, C, false), C, C, type);
        add(b + "attn_output.bias", P[h + "attn.proj.b"]->w.v, C, 1, GGML_F32);
        add(b + "ffn_norm.weight", P[h + "ln2.g"]->w.v, C, 1, GGML_F32);
        add(b + "ffn_gate.weight", rows_of(P[h + "fc1.W"]->w, 0, F, false), C, F, type);
        add(b + "ffn_up.weight", rows_of(P[h + "fc1.W"]->w, F, F, false), C, F, type);
        add(b + "ffn_down.weight", rows_of(P[h + "fc2.W"]->w, 0, C, false), F, C, type);
    }
    add("output_norm.weight", P["lnf.g"]->w.v, C, 1, GGML_F32);
    add("output.weight", rows_of(P["head.W"]->w, 0, V, false), C, V, type);
    return w.save(path, err);
}

// A small modern model whose biases outside attention are zero (GGUF llama/qwen2 have none there).
static std::unique_ptr<Model> gguf_test_model(int vocab, Rng& rng) {
    ModelSpec s;
    s.kind = MODEL_GPT; s.vocab = vocab; s.ctx = 32; s.dim = 32; s.heads = 4; s.nlayer = 2; s.arch = ARCH_MODERN;
    auto m = std::make_unique<Model>(s, 5);
    for (Param* p : m->params()) {
        bool zero = p->name.find(".fc1.b") != std::string::npos || p->name.find(".fc2.b") != std::string::npos || p->name == "head.b";
        for (auto& v : p->w.v) v = zero ? 0.f : v + rng.normal() * 0.15f;
    }
    return m;
}

static int gguf_selftest() {
    int fails = 0;
    Rng rng(11);
    std::string dir = std::filesystem::temp_directory_path().u8string(), err;
    auto tmp = [&](const char* n) { return (std::filesystem::u8path(dir) / n).u8string(); };
    {   // number formats
        std::vector<float> x(256);
        for (auto& v : x) v = rng.normal();
        bool pass = true;
        for (int t : {GGML_F16, GGML_Q8_0, GGML_Q4_0}) {
            std::vector<uint8_t> b;
            std::vector<float> y(x.size());
            ggml_quantize(t, x.data(), (int64_t)x.size(), b);
            ggml_dequantize(t, b.data(), (int64_t)x.size(), y.data());
            double worst = 0;
            for (size_t i = 0; i < x.size(); ++i) worst = std::max(worst, (double)std::fabs(x[i] - y[i]));
            double lim = t == GGML_F16 ? 2e-3 : t == GGML_Q8_0 ? 0.02 : 0.4;
            std::printf("gguf %s round trip: largest error %.4f %s\n", ggml_type_name(t), worst, worst < lim ? "OK" : "FAIL");
            pass = pass && worst < lim;
        }
        if (!pass) ++fails;
    }
    TestVocab num;
    for (int i = 0; i < 40; ++i) { num.toks.push_back("t" + std::to_string(i)); num.types.push_back(1); }
    auto ref = gguf_test_model(40, rng);
    Batch b;
    b.B = 3; b.T = 12;
    for (int i = 0; i < 36; ++i) { b.ids.push_back(rng.below(40)); b.tgt.push_back(rng.below(40)); }
    float ref_loss = ref->step(b, false, 1.f);
    std::vector<int> prompt = {3, 7, 1, 9}, ref_gen;
    {
        Sampling greedy;
        greedy.temperature = 0.f;
        Rng r(1);
        ref->generate(prompt, 20, greedy, r, -1, [&](int t) { ref_gen.push_back(t); return true; });
    }
    std::vector<float> w(36, 1.f);
    static const std::pair<const char*, int> kinds[] = {{"qwen2", GGML_F32}, {"llama", GGML_F32}, {"llama", GGML_Q8_0}};
    for (const auto& [arch, type] : kinds) {
        std::string path = tmp("kor-selftest.gguf");
        bool pass = export_gguf_test(*ref, path, arch, type, num, &err);
        auto g = pass ? GgufModel::load(path, &err) : nullptr;
        float loss = g ? g->lora_loss(b.ids.data(), b.tgt.data(), w.data(), 3, 12, false) : 1e9f;
        std::vector<int> gen;
        if (g) {
            Sampling greedy;
            greedy.temperature = 0.f;
            Rng r(1);
            g->generate(prompt, 20, greedy, r, [&](int t) { gen.push_back(t); return true; });
        }
        bool exact = type == GGML_F32;
        pass = g && std::fabs(loss - ref_loss) < (exact ? 2e-4 : 0.03) && (!exact || gen == ref_gen);
        std::printf("gguf %s %s: loss %.5f vs %.5f, greedy text %s %s %s\n", arch, ggml_type_name(type), loss, ref_loss,
                    exact ? (gen == ref_gen ? "same" : "DIFFERENT") : "-", pass ? "OK" : "FAIL", err.c_str());
        if (!pass) ++fails;
        if (g && std::string(arch) == "qwen2") {
            // LoRA: gradients against finite differences, then a save / load round trip
            LoraConfig lc;
            lc.rank = 2;
            lc.mlp = true;
            g->lora_init(lc);
            auto ps = g->lora_params();
            for (Param* p : ps)
                for (auto& v : p->w.v) v = rng.normal() * 0.3f;
            for (Param* p : ps) p->g.zero();
            double l0 = g->lora_loss(b.ids.data(), b.tgt.data(), w.data(), 3, 12, true), worst = 0;
            for (int k = 0; k < 24; ++k) {
                Param* p = ps[rng.below((int)ps.size())];
                size_t i = (size_t)rng.below((int)p->w.v.size());
                float keep = p->w.v[i], eps = 2e-3f;
                p->w.v[i] = keep + eps;
                double up = g->lora_loss(b.ids.data(), b.tgt.data(), w.data(), 3, 12, false);
                p->w.v[i] = keep - eps;
                double dn = g->lora_loss(b.ids.data(), b.tgt.data(), w.data(), 3, 12, false);
                p->w.v[i] = keep;
                double num_g = (up - dn) / (2 * eps), an = p->g.v[i];
                worst = std::max(worst, std::fabs(num_g - an) / std::max(1e-2, std::fabs(num_g) + std::fabs(an)));
            }
            std::printf("gguf LoRA gradient check: worst relative error %.4f %s\n", worst, worst < 0.05 ? "OK" : "FAIL");
            if (worst >= 0.05) ++fails;
            std::string lp = tmp("kor-selftest-lora.gguf");
            bool ok = g->lora_save(lp, &err);
            auto g2 = ok ? GgufModel::load(path, &err) : nullptr;
            ok = g2 && g2->lora_load(lp, &err);
            double l1 = ok ? g2->lora_loss(b.ids.data(), b.tgt.data(), w.data(), 3, 12, false) : 1e9;
            bool same = ok && std::fabs(l1 - l0) < 1e-5;
            std::printf("gguf LoRA adapter save / load: loss %.5f vs %.5f %s %s\n", l1, l0, same ? "OK" : "FAIL", err.c_str());
            if (!same) ++fails;
            std::error_code ec;
            std::filesystem::remove(std::filesystem::u8path(lp), ec);
        }
        std::error_code ec;
        std::filesystem::remove(std::filesystem::u8path(path), ec);
    }
    {   // byte-level BPE (qwen2 split), special tokens, ChatML
        TestVocab tv;
        tv.tmpl = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}";
        std::vector<std::string> bytes(256);
        {
            int n = 0;
            for (int c = 0; c < 256; ++c) {
                bool keep = (c >= '!' && c <= '~') || (c >= 0xA1 && c <= 0xAC) || (c >= 0xAE && c <= 0xFF);
                uint32_t cp = keep ? (uint32_t)c : (uint32_t)(256 + n++);
                std::string s;
                if (cp < 0x80) s += (char)cp;
                else { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
                bytes[c] = s;
            }
        }
        for (auto& s : bytes) { tv.toks.push_back(s); tv.types.push_back(1); }
        std::string G = bytes[' '];
        static const char* const merged[] = {"Gw", "or", "ld", "Gwor", "Gworld", "He", "ll", "Hell", "Hello"};
        for (const char* t : merged) {
            std::string s = t;
            if (s[0] == 'G') s = G + s.substr(1);
            tv.toks.push_back(s);
            tv.types.push_back(1);
        }
        tv.merges = {G + " w", "o r", "l d", G + "w or", G + "wor ld", "H e", "l l", "He ll", "Hell o"};
        tv.toks.push_back("<|im_start|>");
        tv.toks.push_back("<|im_end|>");
        tv.types.insert(tv.types.end(), 2, 3);
        int V = (int)tv.toks.size();
        auto m = gguf_test_model(V, rng);
        std::string path = tmp("kor-selftest-bpe.gguf");
        auto g = export_gguf_test(*m, path, "qwen2", GGML_F32, tv, &err) ? GgufModel::load(path, &err) : nullptr;
        bool pass = false;
        if (g) {
            auto id = [&](const std::string& s) {
                for (int i = 0; i < V; ++i) if (tv.toks[i] == s) return i;
                return -1;
            };
            std::vector<int> ids = g->encode("Hellod world<|im_end|>12", true), want = {id("Hello"), id("d"), id(G + "world"), id("<|im_end|>"), id("1"), id("2")};
            std::vector<LlmMsg> hi(1);
            hi[0].role = "user";
            hi[0].text = "Hi";
            std::string chat = g->chat_text(hi);
            pass = ids == want && g->decode(ids) == "Hellod world12" && g->is_end(id("<|im_end|>")) &&
                   chat == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n" && g->chat_ids(hi)[0] == id("<|im_start|>") &&
                   g->decode(g->encode("Çağrı 2024, şimdi!\n\n  ok", false)) == "Çağrı 2024, şimdi!\n\n  ok";
            LoraData ld = lora_data(*g, "Q: Hi\nA: Hello\nQ: Hello\nA: Hi\n", "Q: ", "A: ");
            pass = pass && ld.chat && ld.ex.size() == 2 && ld.w[0].front() == 0.f && ld.w[0].back() == 1.f;
        }
        std::printf("gguf BPE tokenizer, special tokens, chat format %s %s\n", pass ? "OK" : "FAIL", err.c_str());
        if (!pass) ++fails;
        std::error_code ec;
        std::filesystem::remove(std::filesystem::u8path(path), ec);
    }
    {   // sentencepiece with byte fallback
        TestVocab tv;
        tv.model = "llama";
        tv.pre = "default";
        auto addt = [&](const std::string& s, int type, float score) { tv.toks.push_back(s); tv.types.push_back(type); tv.scores.push_back(score); };
        addt("<unk>", 2, 0);
        addt("<s>", 3, 0);
        addt("</s>", 3, 0);
        for (int c = 0; c < 256; ++c) {
            char n[8];
            std::snprintf(n, sizeof n, "<0x%02X>", c);
            addt(n, 6, 0);
        }
        std::string U = "\xE2\x96\x81";
        addt(U + "Hello", 1, -1);
        addt(U + "Hel", 1, -2);
        addt(U + "H", 1, -3);
        addt("el", 1, -4);
        addt("lo", 1, -5);
        addt(U, 1, -6);
        int V = (int)tv.toks.size();
        auto m = gguf_test_model(V, rng);
        std::string path = tmp("kor-selftest-spm.gguf");
        auto g = export_gguf_test(*m, path, "llama", GGML_F32, tv, &err) ? GgufModel::load(path, &err) : nullptr;
        bool pass = false;
        if (g) {
            std::vector<int> ids = g->encode("Hello Hi", true), want = {1, V - 6, V - 4, 3 + 'i'};
            pass = ids == want && g->decode(ids) == " Hello Hi" && g->is_end(2);
        }
        std::printf("gguf sentencepiece tokenizer %s %s\n", pass ? "OK" : "FAIL", err.c_str());
        if (!pass) ++fails;
        std::error_code ec;
        std::filesystem::remove(std::filesystem::u8path(path), ec);
    }
    return fails;
}

static int selftest(bool quick) {
    int fails = 0;
    set_threads(4);
    std::printf("kernels: %s\n", K().name);
    Rng rng(3);
    {   // MLP classifier with every layer type
        ModelSpec s;
        s.kind = MODEL_MLP; s.in_dim = 5; s.out_dim = 3;
        int types[] = {L_DENSE, L_LAYERNORM, L_GELU, L_DENSE, L_TANH, L_DENSE, L_SIGMOID, L_DENSE, L_RELU};
        for (int t : types) { LayerSpec l; l.type = t; l.units = 12; s.layers.push_back(l); }
        Model m(s);
        Batch b;
        b.x.resize(7, 5);
        for (auto& v : b.x.v) v = rng.normal();
        for (int i = 0; i < 7; ++i) b.tgt.push_back(rng.below(3));
        if (gradcheck(m, b, "mlp") >= 0.08) ++fails;
        m.enable_lora(2);
        for (Param* p : m.params())
            if (p->trainable) for (auto& v : p->w.v) v = rng.normal() * 0.3f;
        if (gradcheck(m, b, "mlp+lora") >= 0.08) ++fails;
    }
    {   // regression head
        ModelSpec s;
        s.kind = MODEL_MLP; s.in_dim = 4; s.out_dim = 1; s.regression = true;
        LayerSpec l; l.units = 8; s.layers.push_back(l);
        l.type = L_TANH; s.layers.push_back(l);
        Model m(s);
        Batch b;
        b.x.resize(6, 4); b.y.resize(6, 1);
        for (auto& v : b.x.v) v = rng.normal();
        for (auto& v : b.y.v) v = rng.normal();
        if (gradcheck(m, b, "regression") >= 0.08) ++fails;
    }
    for (int arch : {ARCH_CLASSIC, ARCH_MODERN}) {   // transformer
        const bool modern = arch == ARCH_MODERN;
        std::printf("-- %s transformer\n", modern ? "modern (RMSNorm, RoPE, SwiGLU)" : "classic");
        ModelSpec s;
        s.kind = MODEL_GPT; s.vocab = 11; s.ctx = 6; s.dim = 16; s.heads = 2; s.nlayer = 2; s.arch = arch;
        Model m(s);
        for (Param* p : m.params()) for (auto& v : p->w.v) v += rng.normal() * 0.15f;  // leave the flat init region
        Batch b;
        b.B = 3; b.T = 6;
        for (int i = 0; i < 18; ++i) { b.ids.push_back(rng.below(11)); b.tgt.push_back(rng.below(11)); }
        if (gradcheck(m, b, modern ? "modern" : "transformer") >= 0.08) ++fails;
        if (modern) {
            size_t n = m.param_count();
            bool pass = n == estimate_params(s);
            std::printf("modern parameter count %zu vs estimate %zu %s\n", n, estimate_params(s), pass ? "OK" : "FAIL");
            if (!pass) ++fails;
        }
        {   // recomputing the blocks in the backward pass (with dropout) changes nothing
            Model ma(s, 77), mb(s, 77);
            ma.set_dropout(0.2f);
            mb.set_dropout(0.2f);
            mb.recompute = true;
            ma.zero_grad();
            mb.zero_grad();
            float la = ma.step(b, true, 1.f), lb = mb.step(b, true, 1.f);
            la += ma.step(b, true, 1.f);  // a second step: the dropout state goes on the same way
            lb += mb.step(b, true, 1.f);
            double worst = 0;
            auto pa = ma.params(), pb = mb.params();
            for (size_t i = 0; i < pa.size(); ++i)
                for (size_t k = 0; k < pa[i]->g.v.size(); ++k) worst = std::max(worst, (double)std::fabs(pa[i]->g.v[k] - pb[i]->g.v[k]));
            bool pass = la == lb && worst == 0;
            std::printf("recompute in the backward pass: loss %.5f vs %.5f, largest gradient difference %.1e %s\n", la, lb, worst,
                        pass ? "OK" : "FAIL");
            if (!pass) ++fails;
        }

        // Checkpoint round trip (full and half precision) and LoRA merge.
        m.vocab = {"a", "b"};
        std::string err, path = "kor_selftest.tmp";
        double l0 = m.step(b, false, 1.f);
        for (int half = 0; half < 2; ++half) {
            bool ok = m.save(path, half != 0, &err);
            auto m2 = ok ? Model::load(path, &err) : nullptr;
            double l1 = m2 ? m2->step(b, false, 1.f) : 1e9;
            bool pass = m2 && std::fabs(l1 - l0) < (half ? 2e-2 : 1e-6);
            std::printf("checkpoint %s round trip: loss %.5f vs %.5f %s %s\n", half ? "f16" : "f32", l0, l1, pass ? "OK" : "FAIL", err.c_str());
            if (!pass) ++fails;
        }
        std::remove(path.c_str());
        m.enable_lora(4);
        for (Param* p : m.params())
            if (p->trainable) for (auto& v : p->w.v) v = rng.normal() * 0.2f;
        if (gradcheck(m, b, modern ? "modern+lora" : "gpt+lora") >= 0.08) ++fails;
        double before = m.step(b, false, 1.f);
        m.merge_lora();
        double after = m.step(b, false, 1.f);
        bool pass = std::fabs(before - after) < 1e-3;
        std::printf("lora merge: loss %.5f vs %.5f %s\n", before, after, pass ? "OK" : "FAIL");
        if (!pass) ++fails;

        // Cached generation must match recomputing the whole window, also past the context length.
        m.vocab.resize(11, "x");
        Rng r1(5), r2(5);
        std::vector<int> ids = {1, 2, 3}, slow;
        std::vector<int> fast = m.generate(ids, 3, 0.f, r1);  // 3 + 3 tokens fill the context of 6
        for (int i = 0; i < 3; ++i) { int t = m.next_token(ids, 0.f, r2); slow.push_back(t); ids.push_back(t); }
        pass = fast == slow;
        std::printf("cached generation %s\n", pass ? "OK" : "FAIL");
        if (!pass) ++fails;
        fast = m.generate({1}, 40, 1.f, r1);
        pass = fast.size() == 40;
        std::printf("cached generation past the context %s\n", pass ? "OK" : "FAIL");
        if (!pass) ++fails;

        // Fine-tuning with new characters keeps the known ones unchanged.
        m.vocab = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"};
        double l2 = m.step(b, false, 1.f);
        int added = m.extend_vocab({"a", "z", "y"});
        double l3 = m.step(b, false, 1.f);
        pass = added == 2 && m.spec.vocab == 13 && std::fabs(l3 - l2) < 0.05;
        std::printf("vocabulary extension: +%d, loss %.5f vs %.5f %s\n", added, l2, l3, pass ? "OK" : "FAIL");
        if (!pass) ++fails;
    }
    {   // media files
        Image img;
        img.w = 7; img.h = 5;
        for (int i = 0; i < 7 * 5 * 3; ++i) img.rgb.push_back((uint8_t)(i * 37));
        std::string err, path = "kor_selftest.png";
        Image back;
        bool pass = save_png(path, img, &err) && load_image(path, back, &err) && back.w == 7 && back.h == 5 && back.rgb == img.rgb;
        std::remove(path.c_str());
        std::printf("png round trip %s %s\n", pass ? "OK" : "FAIL", err.c_str());
        if (!pass) ++fails;
        Sound snd;
        snd.rate = 8000;
        for (int i = 0; i < 800; ++i) snd.samples.push_back(0.5f * std::sin(i * 0.1f));
        Sound sb;
        path = "kor_selftest.wav";
        pass = save_wav(path, snd, &err) && load_wav(path, sb, &err) && sb.rate == 8000 && sb.samples.size() == 800 &&
               std::fabs(sb.samples[17] - snd.samples[17]) < 1e-3f;
        std::remove(path.c_str());
        std::printf("wav round trip %s %s\n", pass ? "OK" : "FAIL", err.c_str());
        if (!pass) ++fails;
        float worst = 0;
        for (int i = 0; i <= 100; ++i) {
            float x = -1.f + i * 0.02f;
            worst = std::max(worst, std::fabs(mulaw_decode(mulaw_encode(x)) - x));
        }
        pass = worst < 0.04f;
        std::printf("mu-law round trip: worst error %.4f %s\n", worst, pass ? "OK" : "FAIL");
        if (!pass) ++fails;
    }
    {   // subword tokens: learned pieces shorten the text and decode back to it
        std::string text = "merhaba dünya\nmerhaba kor\nmerhaba dünya, merhaba kor\n";
        auto chars = utf8_chars(text);
        auto v = learn_subwords(text, 40);
        std::vector<int> ids = tokenize(text, v);
        std::string back;
        for (int t : ids) back += v[t];
        bool no_nl_merge = true;
        for (auto& p : v) no_nl_merge &= p == "\n" || p.find('\n') == std::string::npos;
        bool pass = back == text && ids.size() < chars.size() / 2 && no_nl_merge;
        std::printf("subword tokens: %zu pieces, %zu chars -> %zu tokens %s\n", v.size(), chars.size(), ids.size(), pass ? "OK" : "FAIL");
        if (!pass) ++fails;
    }
    {   // the tokenizer gives the longest match over whole characters, like a plain search
        std::string text = "ab\r\nabc ğüşabcd\r\rab\xC3" "A x\xE2\x82\nabab\xF0\x9F\x99\x82 q?";
        std::vector<std::string> v = {"a", "b", "ab", "abc", "abcd", "\n", " ", "ğ", "ü", "ş", "ğü", "x", "\xC3" "A", "q",
                                      "b\na", "\xF0\x9F\x99\x82", "ab\xF0\x9F\x99\x82", "c", "d", "A"};
        auto chars = utf8_chars(text);
        std::vector<int> ref;
        for (size_t i = 0; i < chars.size();) {
            int best = -1;
            size_t best_len = 1;
            std::string piece;
            for (size_t l = 1; i + l <= chars.size(); ++l) {
                piece += chars[i + l - 1];
                for (size_t k = 0; k < v.size(); ++k)
                    if (v[k] == piece && best_len <= l) { best = (int)k; best_len = l; break; }
            }
            if (best >= 0) ref.push_back(best);
            i += best_len;
        }
        bool pass = tokenize(text, v) == ref;
        std::printf("tokenizer: %zu tokens %s\n", ref.size(), pass ? "OK" : "FAIL");
        if (!pass) ++fails;
    }
    if (!quick) {   // crash-safe training: a run stopped halfway and resumed ends like an uninterrupted one
        std::string err;
        float diff = resume_check(false, err);
        bool pass = diff >= 0 && diff < 1e-4f;
        std::printf("resume after stop: weight difference %.2g %s %s\n", diff, pass ? "OK" : "FAIL", err.c_str());
        if (!pass) ++fails;
    }
    {   // sampling: top-k, top-p and the repeat penalty on fixed logits
        ModelSpec sp;
        sp.kind = MODEL_GPT; sp.dim = 8; sp.heads = 2; sp.nlayer = 1; sp.ctx = 8; sp.vocab = 5;
        Model m(sp);
        const float z[5] = {2.f, 1.9f, 0.5f, -1.f, 1.f};
        Rng r(5);
        Sampling s;
        s.temperature = 1.f;
        bool pass = true;
        int counts[5] = {0};
        s.top_k = 2;
        for (int i = 0; i < 400; ++i) ++counts[m.sample(z, s, nullptr, 0, r)];
        pass &= counts[0] > 0 && counts[1] > 0 && counts[2] + counts[3] + counts[4] == 0;
        s.top_k = 0;
        s.top_p = 0.3f;
        for (int i = 0; i < 100; ++i) pass &= m.sample(z, s, nullptr, 0, r) == 0;
        s.top_p = 1.f;
        s.temperature = 0.f;
        s.repeat_penalty = 1.5f;
        const int recent[2] = {0, 0};
        pass &= m.sample(z, s, recent, 2, r) == 1 && m.sample(z, s, recent, 2, r, 0) == 0;
        std::printf("sampling (top-k, top-p, repeat penalty) %s\n", pass ? "OK" : "FAIL");
        if (!pass) ++fails;
    }
    {   // the model's name: language guess, identity lines in front of the training text, kept in the file
        bool pass = text_language("Kullanıcı: Nasılsın?\nAsistan: İyiyim, teşekkürler.\n") == LANG_TEXT_TR &&
                    text_language("User: How are you?\nAssistant: Fine, thanks.\n") == LANG_TEXT_EN &&
                    text_language("Пользователь: Как дела?\nАссистент: Хорошо.\n") == LANG_TEXT_RU;
        std::string block = identity_dialogue("Atlas", "Ada", "Kullanıcı: ", "Asistan: ", LANG_TEXT_TR);
        pass = pass && block.find("Kullanıcı: Adın ne?\nAsistan: Benim adım Atlas.\n") == 0 &&
               block.find("Beni Ada yaptı.") != std::string::npos && block.find('{') == std::string::npos;
        std::string path = (std::filesystem::temp_directory_path() / "kor-selftest-name.txt").u8string(), err;
        {
            std::ofstream f(upath(path), std::ios::binary);
            for (int i = 0; i < 400; ++i) f << "Kullanıcı: Merhaba\nAsistan: Selam, nasılsın?\n";
        }
        Dataset d;
        if (d.load_text(path, &err)) {
            ModelSpec sp;
            sp.kind = MODEL_GPT; sp.dim = 16; sp.heads = 2; sp.nlayer = 1; sp.ctx = 16; sp.vocab = (int)d.vocab.size();
            Model m(sp);
            m.vocab = d.vocab;
            m.meta["name"] = "Atlas";
            size_t n0 = d.n_train, v0 = m.vocab.size();
            m.extend_vocab(text_chars(block));
            d.vocab = m.vocab;
            int reps = d.add_training_text(block, 0.01);
            std::string front, want = "Kullanıcı: Adın ne?\nAsistan: Benim adım Atlas.";
            for (size_t i = 0; i < 60; ++i) front += d.vocab[d.ids[i]];
            pass = pass && reps >= 3 && m.vocab.size() > v0 && d.n_train > n0 && front.compare(0, want.size(), want) == 0 &&
                   d.layout.starts.size() > 400;
            std::string mp = path + ".kor";
            auto back = m.save(mp, true, &err) ? Model::load(mp, &err) : nullptr;
            pass = pass && back && back->meta["name"] == "Atlas";
            std::error_code ec;
            std::filesystem::remove(upath(mp), ec);
        } else pass = false;
        std::error_code ec;
        std::filesystem::remove(upath(path), ec);
        std::printf("model name and identity lines %s\n", pass ? "OK" : "FAIL");
        if (!pass) ++fails;
    }
    {   // half-float conversion
        float vals[] = {0.f, 1.f, -2.5f, 0.1f, 65504.f, 1e-5f, 3.14159f};
        bool pass = true;
        for (float v : vals) {
            float r = half_to_float(float_to_half(v));
            if (std::fabs(r - v) > std::max(1e-6f, std::fabs(v) * 1e-3f)) pass = false;
        }
        std::printf("half float conversion %s\n", pass ? "OK" : "FAIL");
        if (!pass) ++fails;
    }
    {
        std::string out, err;
        ImportReport ir;
        bool pass = import_jsonl("{\"messages\":[{\"role\":\"system\",\"content\":\"x\"},{\"role\":\"user\",\"content\":\"Hi\\nthere\"},"
                                 "{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"Hello\"}]}]}\n"
                                 "{\"instruction\":\"Add\",\"input\":\"1+1\",\"output\":\"2\"}\nbroken\n",
                                 "Q: ", "A: ", out, ir, &err) &&
                    out == "Q: Hi there\nA: Hello\nQ: Add 1+1\nA: 2\n" && ir.dialogues == 2 && ir.skipped == 1;
        CleanReport cr;
        std::string c = clean_text("Q: a  b \nA: c\n\nQ: a b\nA: c\nQ: lost\nQ: d\nA: e\n", "Q: ", "A: ", CleanOptions(), cr);
        pass = pass && c == "Q: a b\nA: c\nQ: d\nA: e\n" && cr.duplicates == 1 && cr.broken == 1 && cr.empty == 1;
        std::printf("dataset tools (JSONL import, cleaning) %s\n", pass ? "OK" : "FAIL");
        if (!pass) ++fails;
    }
    std::printf("-- GGUF models\n");
    fails += gguf_selftest();
    std::printf(fails ? "SELFTEST FAILED (%d)\n" : "SELFTEST PASSED\n", fails);
    return fails ? 1 : 0;
}

// Question tests from a file, using the dialogue tags of `data` if it has them.
static bool load_tests(const std::string& path, const Dataset* data, std::vector<Probe>& out) {
    std::string text, qt, at;
    if (!read_text(path, text)) return false;
    if (data) data->dialogue_tags(qt, at);
    out = parse_tests(text, qt, at);
    return true;
}

static int train(const std::vector<std::string>& a) {
    std::string path = a[2], out = flag(a, "--out"), tests = flag(a, "--tests"), log_path = flag(a, "--log");
    int steps = std::atoi(flag(a, "--steps", "0").c_str()), patience = std::atoi(flag(a, "--patience", "0").c_str());
    float target = (float)std::atof(flag(a, "--target", "0").c_str());
    int kind = has_flag(a, "--table") ? DATA_TABLE : has_flag(a, "--text") ? DATA_TEXT : 0;
    int media = has_flag(a, "--images") ? MEDIA_IMAGE : has_flag(a, "--audio") ? MEDIA_AUDIO : MEDIA_TEXT;
    std::error_code ec;
    if (!kind && media == MEDIA_TEXT) {
        if (std::filesystem::is_directory(upath(path), ec)) {
            // A folder: images if it has any, otherwise sound.
            for (std::filesystem::recursive_directory_iterator it(upath(path), ec), end; !ec && it != end; it.increment(ec)) {
                std::string f = it->path().u8string();
                if (is_image_file(f)) { media = MEDIA_IMAGE; break; }
                if (is_wav_file(f)) media = MEDIA_AUDIO;
            }
        } else if (is_wav_file(path)) {
            media = MEDIA_AUDIO;
        }
    }
    if (media != MEDIA_TEXT) kind = DATA_TEXT;
    if (!kind) {
        std::string ext = path.size() > 4 ? path.substr(path.size() - 4) : "";
        kind = (ext == ".csv" || ext == ".tsv") ? DATA_TABLE : DATA_TEXT;
    }
    std::string err;
    // Crash safety: the run is saved every minute and continues with --resume.
    bool use_ckpt = !has_flag(a, "--no-checkpoint");
    std::string ckpt = flag(a, "--checkpoint");
    if (ckpt.empty()) ckpt = (out.empty() ? std::string("kor-train") : out) + ".autosave";
    CheckpointInfo info;
    std::shared_ptr<Model> resumed;
    if (use_ckpt && has_checkpoint(ckpt)) {
        if (has_flag(a, "--resume")) {
            if (!read_checkpoint(ckpt, info, &err)) { std::printf("error: %s: %s\n", ckpt.c_str(), err.c_str()); return 1; }
            resumed = Model::load(ckpt, &err);
            if (!resumed) { std::printf("error: %s: %s\n", ckpt.c_str(), err.c_str()); return 1; }
            std::printf("resuming the saved training from step %d of %d (%s)\n", info.step, info.cfg.steps, ckpt.c_str());
        } else if (!has_flag(a, "--fresh")) {
            std::printf("error: an unfinished training is saved in %s\n"
                        "       add --resume to continue it, or --fresh to start over (the saved one is replaced)\n", ckpt.c_str());
            return 1;
        }
    } else if (has_flag(a, "--resume")) {
        std::printf("error: no unfinished training in %s (use the same --out or --checkpoint as before)\n", ckpt.c_str());
        return 1;
    }
    // Fine-tuning: start from a saved model instead of a new one. Images and
    // sound are read in the format of that model.
    std::shared_ptr<Model> base;
    std::string from = resumed ? std::string() : flag(a, "--from");
    if (resumed) base = resumed;
    if (!from.empty()) {
        base = Model::load(from, &err);
        if (!base) { std::printf("error: %s: %s\n", from.c_str(), err.c_str()); return 1; }
        if (base->spec.lora_rank > 0) base->merge_lora();
        if (base->spec.kind == MODEL_GPT && base->spec.media != MEDIA_TEXT && kind == DATA_TEXT) media = base->spec.media;
    }
    Dataset data;
    bool ok;
    if (kind == DATA_TABLE) ok = data.load_table(path, -1, &err);
    else if (media == MEDIA_IMAGE) {
        int size = std::atoi(flag(a, "--size", "16").c_str()), colors = std::atoi(flag(a, "--colors", "32").c_str());
        bool fixed = base && base->spec.media == MEDIA_IMAGE;
        ok = data.load_images(path, fixed ? base->spec.img_size : size, colors, !has_flag(a, "--no-mirror"), &err,
                              fixed ? &base->vocab : nullptr);
    } else if (media == MEDIA_AUDIO) {
        int rate = std::atoi(flag(a, "--rate", "8000").c_str());
        ok = data.load_audio(path, base && base->spec.media == MEDIA_AUDIO ? base->spec.rate : rate, &err);
    } else {
        ok = data.load_text(path, &err, std::atoi(flag(a, "--subword", "0").c_str()));
    }
    if (!ok) { std::printf("error: %s\n", err.c_str()); return 1; }
    if (media == MEDIA_IMAGE)
        std::printf("images: %d read (%d skipped), %dx%d pixels, %zu colours, %zu tokens\n", data.items, data.skipped,
                    data.img_size, data.img_size, data.vocab.size() - 1, data.ids.size());
    else if (media == MEDIA_AUDIO)
        std::printf("sound: %d WAV files read (%d skipped), %d Hz, %.1f seconds\n", data.items, data.skipped, data.rate,
                    (double)data.ids.size() / data.rate);
    else if (kind == DATA_TEXT)
        std::printf("text: %zu tokens, vocabulary %zu\n", data.ids.size(), data.vocab.size());
    SysInfo sys = detect_system();
    AutoPlan plan = autotune(sys, data, has_flag(a, "--gpu") ? gpu_probe().mem_gb : 0);
    if (steps > 0) { plan.cfg.steps = steps; plan.cfg.eval_every = std::max(10, steps / 10); }
    else plan.cfg.eval_every = std::max(10, plan.cfg.steps / 10);
    auto num = [&](const char* name, auto& v) {
        std::string s = flag(a, name);
        if (!s.empty()) v = (std::remove_reference_t<decltype(v)>)std::atof(s.c_str());
    };
    if (kind == DATA_TEXT) {
        num("--dim", plan.spec.dim);
        num("--layers", plan.spec.nlayer);
        num("--heads", plan.spec.heads);
        num("--ctx", plan.spec.ctx);
        std::string arch = flag(a, "--arch");
        if (arch == "classic") plan.spec.arch = ARCH_CLASSIC;
        else if (arch == "modern") plan.spec.arch = ARCH_MODERN;
        else if (!arch.empty()) { std::printf("error: --arch is classic or modern\n"); return 1; }
        num("--ffn", plan.spec.ffn);
        if (plan.spec.arch == ARCH_MODERN && (plan.spec.dim / std::max(1, plan.spec.heads)) % 2 != 0) {
            std::printf("error: the modern architecture needs an even --dim / --heads\n");
            return 1;
        }
        if (media == MEDIA_IMAGE) plan.spec.ctx = data.img_size * data.img_size + 1;  // one whole image
        if (plan.spec.dim % plan.spec.heads != 0) { std::printf("error: --dim must be a multiple of --heads\n"); return 1; }
        plan.params = estimate_params(plan.spec);
    }
    if (resumed) {
        bool fits = resumed->spec.kind == MODEL_GPT ? kind == DATA_TEXT && resumed->spec.media == media
                                                    : kind == DATA_TABLE && resumed->spec.in_dim == data.n_feat;
        if (!fits) { std::printf("error: the saved training does not fit this dataset\n"); return 1; }
        if (kind == DATA_TEXT && media == MEDIA_TEXT && data.vocab != resumed->vocab) data.encode_with(resumed->vocab);
        plan.spec = resumed->spec;
        plan.params = estimate_params(plan.spec);
    } else if (base) {
        bool fits = base->spec.kind == MODEL_GPT ? kind == DATA_TEXT && base->spec.media == media
                  : kind == DATA_TABLE && base->spec.in_dim == data.n_feat && base->spec.regression == data.regression &&
                    (data.regression || base->vocab == data.labels);
        if (!fits) { std::printf("error: %s does not fit this dataset\n", from.c_str()); return 1; }
        if (kind == DATA_TEXT && media == MEDIA_TEXT) {
            int added = base->extend_vocab(data.vocab);
            data.encode_with(base->vocab);
            if (added) std::printf("fine-tuning: %d new characters added to the model\n", added);
        }
        plan.spec = base->spec;
        plan.params = estimate_params(plan.spec);
        plan.cfg.lr /= 3;  // a trained model needs smaller steps
        int lora = std::atoi(flag(a, "--lora", "0").c_str());
        if (lora > 0) base->enable_lora(lora);
        std::printf("fine-tuning %s%s\n", from.c_str(), lora > 0 ? (" with LoRA rank " + std::to_string(lora)).c_str() : "");
    }
    num("--lr", plan.cfg.lr);
    num("--batch", plan.cfg.batch);
    num("--threads", plan.cfg.threads);
    num("--seed", plan.cfg.seed);
    num("--align", plan.cfg.align);
    num("--prompt-weight", plan.cfg.prompt_weight);
    num("--dropout", plan.cfg.dropout);
    num("--wd", plan.cfg.weight_decay);
    num("--ema", plan.cfg.ema);
    plan.cfg.gpu = has_flag(a, "--gpu");
    // on the GPU a text model keeps only weights, gradients, moments and a best copy in RAM
    const bool gpu_text = plan.cfg.gpu && gpu_probe().found && plan.spec.kind == MODEL_GPT && plan.spec.lora_rank == 0;
    plan.est_ram_mb = gpu_text ? (double)plan.params * 4 * 5 / (1024.0 * 1024.0)
                               : (double)estimate_train_bytes(plan.spec, plan.cfg.batch) / (1024.0 * 1024.0);
    std::printf("auto plan: %zu parameters, ~%.0f MB RAM, batch %d x%d, %d steps, %d threads, kernels %s\n",
                plan.params, plan.est_ram_mb, plan.cfg.batch, plan.cfg.accum, plan.cfg.steps, plan.cfg.threads, K().name);
    if (kind == DATA_TEXT)
        std::printf("model: %s, dim %d, %d layers, %d heads, context %d; lr %g; aligned rows %.0f%%, question weight %g, dropout %g, weight decay %g, ema %g\n",
                    plan.spec.arch == ARCH_MODERN ? "modern (RMSNorm, RoPE, SwiGLU)" : "classic", plan.spec.dim, plan.spec.nlayer,
                    plan.spec.heads, plan.spec.ctx, plan.cfg.lr, plan.cfg.align * 100,
                    plan.cfg.prompt_weight, plan.cfg.dropout, plan.cfg.weight_decay, plan.cfg.ema);
    if (media == MEDIA_IMAGE) std::printf("image model: %dx%d pixels, %zu colours\n", data.img_size, data.img_size, data.vocab.size() - 1);
    if (media == MEDIA_AUDIO) std::printf("sound model: %d Hz, 256 levels (mu-law)\n", data.rate);
    auto model = base ? base : std::make_shared<Model>(plan.spec);
    model->recompute = has_flag(a, "--recompute");
    if (!base) {
        if (kind == DATA_TEXT) model->vocab = data.vocab;
        else { model->vocab = data.labels; model->norm_mean = data.mean; model->norm_std = data.stdv; }
    }
    if (kind == DATA_TEXT && media == MEDIA_TEXT) {
        if (!tests.empty()) {
            if (!load_tests(tests, &data, plan.cfg.probes)) { std::printf("error: cannot read %s\n", tests.c_str()); return 1; }
            std::printf("%zu question tests from %s\n", plan.cfg.probes.size(), tests.c_str());
        } else {
            plan.cfg.probes = data.auto_probes(12);
            if (!plan.cfg.probes.empty()) std::printf("dialogue data detected: %zu question tests\n", plan.cfg.probes.size());
        }
        if (int held = data.hold_out(plan.cfg.probes))
            std::printf("test questions held out: %d lines removed from the training data\n", held);
        std::string name = flag(a, "--name"), maker = flag(a, "--maker"), qt, at;
        if (!name.empty()) model->meta["name"] = name;
        if (!maker.empty()) model->meta["maker"] = maker;
        auto ni = model->meta.find("name"), mi = model->meta.find("maker");
        if (ni != model->meta.end() && !has_flag(a, "--no-teach-name") && model->spec.lora_rank == 0 && data.dialogue_tags(qt, at)) {
            std::string block = identity_dialogue(ni->second, mi == model->meta.end() ? "" : mi->second, qt, at, text_language(data.raw));
            model->extend_vocab(text_chars(block));
            data.vocab = model->vocab;
            int reps = data.add_training_text(block, 0.01);
            std::printf("teaching the name \"%s\": its question-answer lines were added %d times\n", ni->second.c_str(), reps);
        }
        if (data.n_train < (size_t)plan.spec.ctx + 2) {
            std::printf("error: too little training text is left after removing the test questions\n");
            return 1;
        }
        plan.cfg.probe_every = std::max(1, plan.cfg.steps / 10);
        plan.cfg.probe_target = std::max(0.f, std::min(1.f, target));
    }
    plan.cfg.patience = patience;
    if (!add_skills(a, path, data, plan.cfg)) return 1;
    if (resumed) {
        // The settings of the saved run; tests and skills come from this command.
        TrainConfig c = info.cfg;
        c.probes = plan.cfg.probes;
        c.hooks = plan.cfg.hooks;
        c.threads = plan.cfg.threads;
        c.gpu = plan.cfg.gpu;
        plan.cfg = c;
    }
    if (use_ckpt) {
        plan.cfg.checkpoint = ckpt;
        plan.cfg.resume = resumed != nullptr;
        num("--checkpoint-every", plan.cfg.checkpoint_every);
        std::printf("autosave: %s every %.0f s (continue later with --resume)\n", ckpt.c_str(), plan.cfg.checkpoint_every);
    }
    std::ofstream log;
    if (!log_path.empty()) {
        log.open(upath(log_path), std::ios::binary);
        if (!log) { std::printf("error: cannot write %s\n", log_path.c_str()); return 1; }
    }
    auto say = [&](const char* fmt, auto... args) {
        char b[2048];
        std::snprintf(b, sizeof b, fmt, args...);
        std::fputs(b, stdout);
        if (log) log << b;
    };
    Trainer tr;
    if (!tr.start(model, &data, plan.cfg)) { std::printf("error: %s\n", tr.stats().error.c_str()); return 1; }
    std::vector<TrainEvent> events;
    auto print_events = [&] {
        events.clear();
        tr.drain_events(events);
        for (auto& e : events) {
            if (e.kind == EV_START) {
                say("start: %lld trainable parameters, %lld threads, kernels %s\n", e.n1, e.n2, e.text.c_str());
            } else if (e.kind == EV_EVAL) {
                say("step %5d  loss %.4f  val %.4f", e.step, e.loss, e.val_loss);
                if (e.val_acc >= 0) say("  acc %.1f%%", e.val_acc * 100);
                say(e.n1 ? "  *best\n" : "\n");
            } else if (e.kind == EV_PROBES) {
                say("step %5d  question tests %lld/%lld   e.g. %s\n", e.step, e.n1, e.n2, e.text.c_str());
            } else if (e.kind == EV_EARLY_STOP) {
                say("step %5d  early stop: no improvement for %lld evaluations\n", e.step, e.n1);
            } else if (e.kind == EV_TARGET) {
                say("step %5d  question test target reached (%lld/%lld), stopping\n", e.step, e.n1, e.n2);
            } else if (e.kind == EV_RESTORED) {
                if (e.n2 > 0) say("restored the weights from step %d, which passed the most question tests (%lld/%lld)\n", e.step, e.n1, e.n2);
                else say("restored best weights from step %d (val %.4f)\n", e.step, e.val_loss);
            } else if (e.kind == EV_DONE || e.kind == EV_STOPPED) {
                say("%s at step %d after %.1fs\n", e.kind == EV_DONE ? "finished" : "stopped", e.step, e.time);
                if (e.kind == EV_STOPPED && use_ckpt) say("saved: continue with the same command and --resume\n");
            } else if (e.kind == EV_RESUMED) {
                say("continuing at step %d (%.1fs trained before)\n", e.step, e.time);
            } else if (e.kind == EV_CHECKPOINT) {
                say("warning: autosave failed at step %d: %s\n", e.step, e.text.c_str());
            } else if (e.kind == EV_ERROR) {
                say("error: %s\n", e.text.c_str());
            } else if (e.kind == EV_HOOK_LOG) {
                static const char* const mark[] = {"", "+ ", "! ", "x ", ""};
                say("[%s] %s%s\n", e.source.c_str(), mark[e.n1 >= 0 && e.n1 < 5 ? e.n1 : 0], e.text.c_str());
            } else if (e.kind == EV_HOOK_METRIC) {
                say("[%s] step %d  %s = %g\n", e.source.c_str(), e.step, e.text.c_str(), e.loss);
            } else if (e.kind == EV_HOOK_STOP) {
                say("[%s] step %d  stop: %s\n", e.source.c_str(), e.step, e.text.empty() ? "-" : e.text.c_str());
            } else if (e.kind == EV_HOOK_LR) {
                if (e.n1) say("[%s] step %d  fixed learning rate %g\n", e.source.c_str(), e.step, e.loss);
                else say("[%s] step %d  learning rate x%.3g\n", e.source.c_str(), e.step, e.loss);
            } else if (e.kind == EV_HOOK_DATA) {
                if (e.n1) say("[%s] step %d  +%lld characters of training text (%lld added in total)\n", e.source.c_str(), e.step, e.n1, e.n2);
                else say("[%s] step %d  added training text cleared\n", e.source.c_str(), e.step);
            } else if (e.kind == EV_HOOK_BEST) {
                if (e.n1) say("[%s] step %d  kept these weights as the best\n", e.source.c_str(), e.step);
                else say("[%s] step %d  went back to the best weights (step %lld)\n", e.source.c_str(), e.step, e.n2);
            } else if (e.kind == EV_HOOK_SAMPLING) {
                say("[%s] step %d  aligned rows %.0f%%, question weight %.2f\n", e.source.c_str(), e.step, e.loss * 100, e.val_loss);
            } else if (e.kind == EV_HOOK_DROPOUT) {
                say("[%s] step %d  dropout %.2f\n", e.source.c_str(), e.step, e.loss);
            }
        }
        std::fflush(stdout);
        if (log) log.flush();
    };
    g_interrupt = false;
    std::signal(SIGINT, on_interrupt);
    bool stopping = false;
    while (tr.running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        print_events();
        if (g_interrupt && !stopping) {
            stopping = true;
            std::printf("\nstopping and saving...\n");
            tr.request_stop();
        }
    }
    std::signal(SIGINT, SIG_DFL);
    print_events();
    TrainStats st = tr.stats();
    if (st.state == TS_ERROR) { std::printf("error: %s\n", st.error.c_str()); return 1; }
    std::printf("done in %.1fs: train loss %.4f, val loss %.4f", st.elapsed, st.loss, st.val_loss);
    if (st.val_acc >= 0) std::printf(", val accuracy %.1f%%", st.val_acc * 100);
    std::printf(", %.0f %s/s\n", st.items_per_sec, kind == DATA_TEXT ? "tokens" : "samples");    if (kind == DATA_TEXT && media == MEDIA_TEXT) {
        Rng rng(11);
        std::vector<int> ids = {data.ids[0]};
        std::string text;
        for (int i = 0; i < 200; ++i) {
            int t = model->next_token(ids, 0.8f, rng);
            ids.push_back(t);
            text += model->vocab[t];
        }
        std::printf("--- sample ---\n%s\n", text.c_str());
    }
    if (!out.empty()) {
        if (!model->save(out, false, &err)) { std::printf("save error: %s\n", err.c_str()); return 1; }
        std::printf("saved %s\n", out.c_str());
    }
    return 0;
}

static std::shared_ptr<Model> load_text_model(const std::string& path) {
    std::string err;
    auto m = Model::load(path, &err);
    if (!m) std::printf("error: %s\n", err.c_str());
    else if (m->spec.kind != MODEL_GPT) { std::printf("error: %s is not a text model\n", path.c_str()); m.reset(); }
    return m;
}

// kor-cli generate <model.kor> <prompt> [--len N] [--temp T] [--line]
static int generate(const std::vector<std::string>& a) {
    auto m = load_text_model(a[2]);
    if (!m) return 1;
    std::string prompt = a.size() > 3 && a[3].compare(0, 2, "--") != 0 ? a[3] : "";
    int len = std::max(1, std::atoi(flag(a, "--len", "200").c_str()));
    float temp = (float)std::atof(flag(a, "--temp", m->spec.media == MEDIA_TEXT ? "0.8" : "1").c_str());
    bool line = has_flag(a, "--line");
    std::string seed = flag(a, "--seed");
    Rng rng(seed.empty() ? (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count() : std::strtoull(seed.c_str(), nullptr, 10));
    m->gen_on_gpu = has_flag(a, "--gpu");
    if (m->spec.media == MEDIA_IMAGE) {
        std::string out = flag(a, "--out", "generated.png");
        int count = std::max(1, std::min(64, std::atoi(flag(a, "--count", "16").c_str())));
        int cols = (int)std::ceil(std::sqrt((double)count)), rows = (count + cols - 1) / cols;
        int s = m->spec.img_size, scale = std::max(1, 64 / s), cell = s * scale + 2;
        Image grid;
        grid.w = cols * cell;
        grid.h = rows * cell;
        grid.rgb.assign((size_t)grid.w * grid.h * 3, 255);
        for (int k = 0; k < count; ++k) {
            Image img = sample_image(*m, temp, rng);
            for (int y = 0; y < s * scale; ++y)
                for (int x = 0; x < s * scale; ++x)
                    for (int c = 0; c < 3; ++c)
                        grid.rgb[((size_t)(k / cols * cell + 1 + y) * grid.w + k % cols * cell + 1 + x) * 3 + c] =
                            img.rgb[((size_t)(y / scale) * s + x / scale) * 3 + c];
            std::printf("\rimage %d/%d", k + 1, count);
            std::fflush(stdout);
        }
        std::string err;
        if (!save_png(out, grid, &err)) { std::printf("\nerror: %s\n", err.c_str()); return 1; }
        std::printf("\nsaved %s\n", out.c_str());
        return 0;
    }
    if (m->spec.media == MEDIA_AUDIO) {
        std::string out = flag(a, "--out", "generated.wav");
        float seconds = std::max(0.1f, (float)std::atof(flag(a, "--seconds", "3").c_str()));
        Sound snd = sample_sound(*m, seconds, temp, rng, [](float done) {
            std::printf("\r%3.0f%%", done * 100);
            std::fflush(stdout);
            return true;
        });
        std::string err;
        if (!save_wav(out, snd, &err)) { std::printf("\nerror: %s\n", err.c_str()); return 1; }
        std::printf("\rsaved %s (%.1f s)\n", out.c_str(), (double)snd.samples.size() / snd.rate);
        return 0;
    }
    std::vector<int> ids = m->encode(prompt);
    if (ids.empty()) ids.push_back(rng.below(std::max(1, m->spec.vocab)));
    std::string out;
    Sampling smp;
    smp.temperature = temp;
    smp.top_p = (float)std::atof(flag(a, "--top-p", "1").c_str());
    smp.top_k = std::atoi(flag(a, "--top-k", "0").c_str());
    smp.repeat_penalty = (float)std::atof(flag(a, "--repeat", "1").c_str());
    for (int t : m->generate(ids, len, smp, rng, -1, [&](int t) { return !(line && m->vocab[t] == "\n"); })) {
        if (line && m->vocab[t] == "\n") break;
        out += m->vocab[t];
    }
    std::printf("%s%s\n", line ? "" : prompt.c_str(), out.c_str());
    return 0;
}

// kor-cli test <model.kor> <tests.txt> [--data <text>] [--tsv]
static int test(const std::vector<std::string>& a) {
    auto m = load_text_model(a[2]);
    if (!m) return 1;
    Dataset data;
    std::string data_path = flag(a, "--data"), err;
    if (!data_path.empty() && !data.load_text(data_path, &err)) { std::printf("error: %s: %s\n", data_path.c_str(), err.c_str()); return 1; }
    std::vector<Probe> probes;
    if (!load_tests(a[3], data_path.empty() ? nullptr : &data, probes)) { std::printf("error: cannot read %s\n", a[3].c_str()); return 1; }
    bool tsv = has_flag(a, "--tsv");
    int pass = 0;
    for (auto& p : probes) {
        ProbeResult r = run_probe(*m, p);
        pass += r.pass;
        if (tsv) std::printf("%s\t%s\t%s\t%s\n", r.pass ? "PASS" : "FAIL", p.question.c_str(), p.expected.c_str(), r.answer.c_str());
        else std::printf("%s  %s\n      answer:   %s\n      expected: %s\n", r.pass ? "PASS" : "FAIL", p.question.c_str(), r.answer.c_str(), p.expected.c_str());
    }
    if (!tsv) std::printf("passed %d/%zu\n", pass, probes.size());
    return 0;
}

static bool write_text(const std::string& path, const std::string& text) {
    std::ofstream o(upath(path), std::ios::binary);
    o << text;
    return (bool)o;
}

static std::string sibling(const std::string& path, const char* suffix) {
    std::filesystem::path p = upath(path);
    return (p.parent_path() / (p.stem().u8string() + suffix)).u8string();
}

// kor-cli data clean <text> [--out file] [--keep-duplicates] [--keep-empty]
// kor-cli data import <file.jsonl> [--out file] [--q-tag "Kullanıcı: "] [--a-tag "Asistan: "]
static int data_tool(const std::vector<std::string>& a) {
    const std::string& sub = a[2];
    const std::string& in = a[3];
    std::string text;
    if (!read_text(in, text)) { std::printf("error: cannot read %s\n", in.c_str()); return 1; }
    if (sub == "import") {
        std::string out, err;
        ImportReport r;
        if (!import_jsonl(text, flag(a, "--q-tag", "Kullanıcı: "), flag(a, "--a-tag", "Asistan: "), out, r, &err)) {
            std::printf("error: %s\n", err.c_str());
            return 1;
        }
        std::string dst = flag(a, "--out", sibling(in, ".txt").c_str());
        if (!write_text(dst, out)) { std::printf("error: cannot write %s\n", dst.c_str()); return 1; }
        std::printf("%zu records (%s): %zu dialogues, %zu turns, %zu skipped\nwrote %s\n", r.records, r.format.c_str(), r.dialogues,
                    r.turns, r.skipped, dst.c_str());
        return 0;
    }
    if (sub == "clean") {
        Dataset d;
        std::string err, qt, at;
        if (d.load_text(in, &err)) d.dialogue_tags(qt, at);
        CleanOptions o;
        o.dedupe = !has_flag(a, "--keep-duplicates");
        o.drop_empty = !has_flag(a, "--keep-empty");
        CleanReport r;
        std::string out = clean_text(text, qt, at, o, r);
        std::string dst = flag(a, "--out", sibling(in, "-clean.txt").c_str());
        if (!write_text(dst, out)) { std::printf("error: cannot write %s\n", dst.c_str()); return 1; }
        if (!qt.empty()) std::printf("dialogue: \"%s\" / \"%s\"\n", qt.c_str(), at.c_str());
        std::printf("%zu -> %zu lines: %zu duplicates, %zu empty, %zu broken removed, %zu tidied\nwrote %s\n", r.lines_in, r.lines_out,
                    r.duplicates, r.empty, r.broken, r.changed, dst.c_str());
        return 0;
    }
    std::printf("error: unknown data command %s (clean or import)\n", sub.c_str());
    return 1;
}

// Dropout and per-row loss weights cannot be compared with the CPU (other masks),
// so the GPU gradient is checked against finite differences of the GPU loss.
static double gpt_numeric_check(int arch, bool recompute) {
    ModelSpec s;
    s.kind = MODEL_GPT;
    s.vocab = 20; s.ctx = 24; s.dim = 32; s.heads = 4; s.nlayer = 2; s.arch = arch;
    Model m(s, 5);
    m.recompute = recompute;
    const int B = 3, T = 24;
    Rng rng(11);
    std::vector<int> ids, tgt;
    std::vector<float> w;
    for (int i = 0; i < B * T; ++i) { ids.push_back(rng.below(s.vocab)); tgt.push_back(rng.below(s.vocab)); w.push_back(0.2f + rng.uniform()); }
    GpuGpt g;
    if (!m.gpu_layout(g)) return 1;
    // bigger weights than at the start, so every part has a clear gradient
    for (Param* p : g.params)
        for (float& x : p->w.v) x += rng.normal() * 0.3f;
    auto loss_at = [&]() {
        float l = 0;
        m.zero_grad();
        if (!gpu_gpt_step(g, ids.data(), tgt.data(), w.data(), B, T, true, 0.25f, 99, 1.f, &l, nullptr)) return std::nanf("");
        return l;
    };
    loss_at();
    std::vector<std::vector<float>> grads;
    for (Param* p : g.params) grads.push_back(p->g.v);
    double worst = 0;
    for (size_t i = 0; i < g.params.size(); ++i) {
        Param* p = g.params[i];
        // the entry with the largest gradient of each parameter
        size_t k = 0;
        for (size_t q = 1; q < grads[i].size(); ++q)
            if (std::fabs(grads[i][q]) > std::fabs(grads[i][k])) k = q;
        if (std::fabs(grads[i][k]) < 1e-4f) continue;
        float keep = p->w.v[k], h = 1e-2f;
        p->w.v[k] = keep + h;
        double lp = loss_at();
        p->w.v[k] = keep - h;
        double lm = loss_at();
        p->w.v[k] = keep;
        double num = (lp - lm) / (2 * h);
        worst = std::max(worst, std::fabs(num - grads[i][k]) / (std::fabs(grads[i][k]) + 1e-3));
    }
    return worst;
}

static double rel_diff(const std::vector<Param*>& a, const std::vector<Param*>& b, bool grads) {
    double worst = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const std::vector<float>& x = grads ? a[i]->g.v : a[i]->w.v;
        const std::vector<float>& y = grads ? b[i]->g.v : b[i]->w.v;
        double diff = 0, ref = 0;
        for (size_t k = 0; k < x.size(); ++k) {
            diff += std::pow((double)x[k] - y[k], 2);
            ref += std::pow((double)x[k], 2);
        }
        worst = std::max(worst, std::sqrt(diff / std::max(ref, 1e-30)));
    }
    return worst;
}

// A batch split into passes of 3 sequences must give the loss and gradients of the CPU.
static double gpt_split_check(int arch, bool recompute) {
    ModelSpec s;
    s.kind = MODEL_GPT;
    s.vocab = 40; s.ctx = 32; s.dim = 32; s.heads = 4; s.nlayer = 2; s.arch = arch;
    Model cpu(s, 8), gpu(s, 8);
    gpu.recompute = recompute;
    Batch b;
    b.B = 8; b.T = 32;
    Rng rng(21);
    for (int i = 0; i < b.B * b.T; ++i) {
        b.ids.push_back(rng.below(s.vocab));
        b.tgt.push_back(rng.below(s.vocab));
        b.w.push_back(0.2f + rng.uniform());
    }
    cpu.use_gpu = false;
    cpu.zero_grad();
    gpu.zero_grad();
    float l0 = cpu.step(b, true, 0.5f);
    gpu_limit_pass_rows(3);
    float l1 = gpu.step(b, true, 0.5f);
    gpu_limit_pass_rows(0);
    if (!gpu.last_step_on_gpu) return 1;
    return std::max(rel_diff(cpu.params(), gpu.params(), true), (double)std::fabs(l0 - l1));
}

// Training steps with weights, gradients and optimizer on the card must end
// with the weights of the same steps on the CPU (clipping included).
static double resident_check(int opt, int arch) {
    ModelSpec s;
    s.kind = MODEL_GPT;
    s.vocab = 40; s.ctx = 32; s.dim = 32; s.heads = 4; s.nlayer = 2; s.arch = arch;
    Model cpu(s, 9), gpu(s, 9);
    cpu.use_gpu = false;
    Optimizer o(opt, 0.01f);
    const float lr = 3e-3f, clip = 0.5f;
    GpuGpt g;
    if (!gpu.gpu_layout(g) || !gpu_resident_begin(g, opt == OPT_SGD ? 1 : 2, 4, s.ctx)) return 1;
    Rng rng(5);
    double worst = 0;
    for (int t = 1; t <= 4; ++t) {
        Batch b;
        b.B = 4; b.T = s.ctx;
        for (int i = 0; i < b.B * b.T; ++i) { b.ids.push_back(rng.below(s.vocab)); b.tgt.push_back(rng.below(s.vocab)); }
        cpu.zero_grad();
        float l0 = 0, l1 = 0;
        for (int a = 0; a < 2; ++a) l0 += cpu.step(b, true, 0.5f);
        float n0 = clip_gradients(cpu.params(), clip);
        o.step(cpu.params(), lr);
        gpu_resident_zero_grad();
        for (int a = 0; a < 2; ++a) l1 += gpu.step(b, true, 0.5f);
        float n1 = 0;
        if (!gpu.last_step_on_gpu || !gpu_resident_update(opt == OPT_ADAMW, lr, 0.01f, clip, t, &n1)) {
            gpu_resident_end(false);
            return 1;
        }
        worst = std::max({worst, (double)std::fabs(l0 - l1), std::fabs(n0 - n1) / (1.0 + n0)});
    }
    gpu_resident_end(true);
    if (opt == OPT_SGD) return std::max(worst, rel_diff(cpu.params(), gpu.params(), false));
    // AdamW moves every weight by about lr per step, also ones whose gradient is
    // only rounding noise (the key bias): measure in parts of one such step.
    for (size_t i = 0; i < g.params.size(); ++i)
        for (size_t k = 0; k < g.params[i]->w.v.size(); ++k)
            worst = std::max(worst, (double)std::fabs(cpu.params()[i]->w.v[k] - g.params[i]->w.v[k]) / lr);
    return worst;
}

// Greedy generation with the weights and keys/values on the card must pick the
// tokens of the CPU, also past the context (the window starts again). Returns the
// number of differing tokens, -1 when the GPU did not generate.
static int gen_check(int arch) {
    ModelSpec s;
    s.kind = MODEL_GPT;
    s.vocab = 50; s.ctx = 24; s.dim = 64; s.heads = 4; s.nlayer = 2; s.arch = arch;
    Model m(s, 13);
    Rng rng(17);
    for (Param* p : m.params())
        for (float& x : p->w.v) x += rng.normal() * 0.2f;
    Rng r1(1), r2(1);
    std::vector<int> prompt = {3, 1, 4, 1, 5};
    std::vector<int> cpu = m.generate(prompt, 40, 0.f, r1);
    m.gen_on_gpu = true;
    std::vector<int> gpu = m.generate(prompt, 40, 0.f, r2);
    bool used = m.last_gen_on_gpu;
    m.gen_on_gpu = false;
    if (!used || gpu.size() != cpu.size()) return -1;
    int diff = 0;
    for (size_t i = 0; i < cpu.size(); ++i) diff += cpu[i] != gpu[i];
    return diff;
}

// Times forward + backward of a text model on random tokens (no data, nothing saved).
static void step_bench(int dim, int heads, int layers, int ctx, int batch, bool gpu_ok) {
    ModelSpec s;
    s.kind = MODEL_GPT;
    s.vocab = 96; s.ctx = ctx; s.dim = dim; s.heads = heads; s.nlayer = layers;
    Model m(s);
    Batch b;
    b.B = batch; b.T = ctx;
    Rng rng(3);
    for (int i = 0; i < batch * ctx; ++i) { b.ids.push_back(rng.below(s.vocab)); b.tgt.push_back(rng.below(s.vocab)); }
    std::printf("dim %3d, %d layers, ctx %3d, batch %2d (%5.2fM params):", dim, layers, ctx, batch, m.param_count() / 1e6);
    std::fflush(stdout);
    double ms[2] = {0, 0};
    float loss[2] = {0, 0};
    std::vector<std::vector<float>> grads;
    double worst = 0;
    for (int g = 0; g < (gpu_ok ? 2 : 1); ++g) {
        if (g == 1) gpu_use(true);
        m.zero_grad();
        loss[g] = m.step(b, true, 1.f);
        // the gradients of both must agree
        std::vector<Param*> ps = m.params();
        for (size_t i = 0; i < ps.size(); ++i) {
            if (g == 0) { grads.push_back(ps[i]->g.v); continue; }
            double diff = 0, ref = 0;
            for (size_t k = 0; k < grads[i].size(); ++k) {
                diff += std::pow((double)grads[i][k] - ps[i]->g.v[k], 2);
                ref += std::pow((double)grads[i][k], 2);
            }
            worst = std::max(worst, std::sqrt(diff / std::max(ref, 1e-30)));
        }
        m.zero_grad();
        auto t0 = std::chrono::steady_clock::now();
        const int reps = 3;
        for (int r = 0; r < reps; ++r) {
            m.step(b, true, 1.f);
            m.zero_grad();
        }
        ms[g] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1000 / reps;
        if (g == 1) gpu_use(false);
        std::printf("  %s %7.1f ms", g ? "GPU" : "CPU", ms[g]);
    }
    if (gpu_ok)
        std::printf("  %5.1fx faster  (loss %.5f / %.5f, gradients differ by %.1e)%s", ms[0] / ms[1], loss[0], loss[1], worst,
                    worst < 1e-3 && std::fabs(loss[0] - loss[1]) < 1e-3f ? "" : "  WRONG");
    std::printf("\n");
}

// kor-cli gpu: finds the OpenCL GPU, checks its products against the CPU and times both.
// quick leaves out the short training run and the step timings.
static int gpu_check(bool quick) {
    const GpuInfo& g = gpu_probe();
    if (!g.found) { std::printf("no usable GPU: %s\n", g.error.c_str()); return 1; }
    std::printf("GPU: %s (%s, %.1f GB, %s)\n", g.name.c_str(), g.vendor.c_str(), g.mem_gb, g.platform.c_str());
    set_threads(recommended_threads(detect_system()));
    const Kernels& cpu = cpu_kernels();
    Rng rng(7);
    const char* names[] = {"mm   ", "mm_ta", "mm_tb"};
    // x, y, z in the argument order of each product; the last ones are shapes of a text model step.
    const int shapes[][3] = {{67, 45, 83}, {129, 257, 63}, {512, 512, 512}, {2048, 256, 768}, {2048, 768, 256}};
    bool all_ok = true;
    std::printf("\n%-6s %-16s %10s %10s %9s %10s\n", "", "size", "CPU ms", "GPU ms", "speed", "max error");
    for (const auto& s : shapes)
        for (int v = 0; v < 3; ++v) {
            int x = s[0], y = s[1], z = s[2];
            size_t na = (size_t)x * y, nb = v == 0 ? (size_t)y * z : (size_t)x * z, nc = v == 1 ? (size_t)y * z : (size_t)x * z;
            if (v == 2) nb = (size_t)z * y;
            std::vector<float> A(na), B(nb), C0(nc), C1, C2;
            for (float& f : A) f = rng.normal();
            for (float& f : B) f = rng.normal();
            for (float& f : C0) f = rng.normal();
            auto time_it = [&](bool gpu, std::vector<float>& C) {
                double best = 1e9;
                for (int r = 0; r < 3; ++r) {
                    C = C0;
                    auto t0 = std::chrono::steady_clock::now();
                    std::string err;
                    if (gpu && !gpu_gemm(v, A.data(), B.data(), C.data(), x, y, z, &err)) {
                        std::printf("GPU error: %s\n", err.c_str());
                        return -1.0;
                    }
                    if (!gpu) {
                        if (v == 0) cpu.mm(A.data(), B.data(), C.data(), x, y, z);
                        else if (v == 1) cpu.mm_ta(A.data(), B.data(), C.data(), x, y, z);
                        else cpu.mm_tb(A.data(), B.data(), C.data(), x, y, z);
                    }
                    best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                }
                return best * 1000;
            };
            double tc = time_it(false, C1), tg = time_it(true, C2);
            if (tg < 0) return 1;
            double err = 0;
            for (size_t i = 0; i < nc; ++i) err = std::max(err, (double)std::fabs(C1[i] - C2[i]) / (1.0 + std::fabs(C1[i])));
            bool ok = err < 1e-3;
            all_ok = all_ok && ok;
            char size[32];
            std::snprintf(size, sizeof size, "%dx%dx%d", x, y, z);
            std::printf("%-6s %-16s %10.2f %10.2f %8.2fx %10.1e%s\n", names[v], size, tc, tg, tc / tg, err, ok ? "" : "  WRONG");
        }
    // attention (forward + backward) against a plain double-precision version
    const int ashapes[][4] = {{2, 77, 3, 8}, {3, 130, 4, 16}, {2, 64, 2, 32}};
    for (const auto& s : ashapes) {
        std::string err;
        double e = gpu_attention_check(s[0], s[1], s[2], s[3], &err);
        bool ok = e >= 0 && e < 1e-3;
        all_ok = all_ok && ok;
        if (e < 0) std::printf("attention B%d T%d H%d hd%d: GPU error: %s\n", s[0], s[1], s[2], s[3], err.c_str());
        else std::printf("attention B%d T%d H%d hd%d: max error %.1e%s\n", s[0], s[1], s[2], s[3], e, ok ? "" : "  WRONG");
    }
    std::string msg;
    bool on = gpu_use(true, &msg);
    for (int arch : {ARCH_CLASSIC, ARCH_MODERN}) {
        if (!on) break;
        std::printf("%s architecture:\n", arch == ARCH_MODERN ? "modern (RMSNorm, RoPE, SwiGLU)" : "classic");
        double e = 0;
        bool ok = true;
        for (bool rc : {false, true}) {
            const char* how = rc ? " (blocks recomputed)" : "";
            e = gpt_numeric_check(arch, rc);
            ok = e < 0.05;
            all_ok = all_ok && ok;
            std::printf("text model with dropout and loss weights%s: gradient vs finite differences %.1e%s\n", how, e, ok ? "" : "  WRONG");
            e = gpt_split_check(arch, rc);
            ok = e < 1e-4;
            all_ok = all_ok && ok;
            std::printf("batch split into passes of 3 sequences%s: difference from the CPU %.1e%s\n", how, e, ok ? "" : "  WRONG");
        }
        for (int opt : {OPT_ADAMW, OPT_SGD}) {
            e = resident_check(opt, arch);
            ok = e < (opt == OPT_SGD ? 1e-4 : 1e-2);
            all_ok = all_ok && ok;
            std::printf("4 steps with weights and %s on the GPU: difference from the CPU %.1e%s\n", opt == OPT_SGD ? "SGD" : "AdamW", e,
                        ok ? "" : "  WRONG");
        }
    }
    gpu_use(false);
    for (int arch : {ARCH_CLASSIC, ARCH_MODERN}) {
        int d = gen_check(arch);
        bool ok = d == 0;
        all_ok = all_ok && ok;
        if (d < 0) std::printf("%s text generation on the GPU: did not run  WRONG\n", arch == ARCH_MODERN ? "modern" : "classic");
        else std::printf("%s text generation on the GPU: %d of 40 greedy tokens differ from the CPU%s\n",
                         arch == ARCH_MODERN ? "modern" : "classic", d, ok ? "" : "  WRONG");
    }
    gpu_gen_release();
    if (on && !quick) {
        std::string err;
        float d = resume_check(true, err);
        bool ok = d >= 0 && d < 1e-4f;
        all_ok = all_ok && ok;
        std::printf("training on the GPU stopped halfway and resumed: weight difference %.2g%s %s\n", d, ok ? "" : "  WRONG", err.c_str());
    }
    std::printf("\n%s: %s\n", all_ok ? "results match the CPU" : "RESULTS DIFFER FROM THE CPU", msg.c_str());
    std::printf("training with --gpu %s\n", on ? "will use the GPU" : "stays on the CPU");
    if (quick) return all_ok ? 0 : 1;
    std::printf("\none training step (forward + backward, random tokens):\n");
    step_bench(64, 4, 3, 112, 32, on);
    step_bench(128, 4, 4, 128, 32, on);
    step_bench(256, 8, 6, 256, 16, on);
    return all_ok ? 0 : 1;
}

// A line typed by the user, as UTF-8 (the Windows console needs the wide API for that).
static bool read_line(std::string& out) {
    out.clear();
#if defined(_WIN32)
    HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    if (GetConsoleMode(in, &mode)) {
        std::wstring w;
        wchar_t buf[512];
        DWORD n = 0;
        while (ReadConsoleW(in, buf, 512, &n, nullptr) && n > 0) {
            w.append(buf, n);
            if (w.back() == L'\n') break;
        }
        if (w.empty()) return false;
        while (!w.empty() && (w.back() == L'\n' || w.back() == L'\r')) w.pop_back();
        int len = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
        out.resize(len);
        WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), out.data(), len, nullptr, nullptr);
        return true;
    }
#endif
    if (!std::getline(std::cin, out)) return false;
    if (!out.empty() && out.back() == '\r') out.pop_back();
    return true;
}

static Sampling sampling_flags(const std::vector<std::string>& a, const char* temp, const char* top_p, const char* top_k, const char* rep) {
    Sampling s;
    s.temperature = (float)std::atof(flag(a, "--temp", temp).c_str());
    s.top_p = (float)std::atof(flag(a, "--top-p", top_p).c_str());
    s.top_k = std::atoi(flag(a, "--top-k", top_k).c_str());
    s.repeat_penalty = (float)std::atof(flag(a, "--repeat", rep).c_str());
    return s;
}

// kor-cli gguf info|chat|generate|lora <model.gguf> ...
static int gguf_cmd(const std::vector<std::string>& a) {
    std::string sub = a[2], path = a[3], err;
    int threads = std::atoi(flag(a, "--threads", "0").c_str());
    set_threads(threads > 0 ? threads : recommended_threads(detect_system()));
    int ctx = std::max(64, std::atoi(flag(a, "--ctx", "4096").c_str()));
    auto t0 = std::chrono::steady_clock::now();
    auto m = GgufModel::load(path, &err, ctx);
    if (!m) {
        std::printf("error: %s\n", err.c_str());
        return 1;
    }
    double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto& c = m->cfg;
    std::printf("%s: %s, %.2f B parameters, %s, %d layers, width %d, %d/%d heads, vocabulary %d, context %d (using %d)\n",
                m->name.c_str(), c.arch.c_str(), m->n_params / 1e9, m->quant.c_str(), c.layers, c.dim, c.heads, c.kv_heads,
                c.vocab, c.ctx_train, m->ctx);
    std::printf("weights: %s\nchat format: %s, file %.0f MB, opened in %.2f s\n", m->quant_mix.c_str(), m->chat_format().c_str(),
                m->file_bytes / 1048576.0, load_s);
    for (const std::string& lp : flags(a, "--lora")) {
        if (!m->lora_load(lp, &err)) {
            std::printf("error: %s: %s\n", lp.c_str(), err.c_str());
            return 1;
        }
        std::printf("LoRA adapter %s (rank %d)\n", lp.c_str(), m->lora_cfg.rank);
    }
    if (sub == "info") return 0;
    std::string seed = flag(a, "--seed");
    Rng rng(seed.empty() ? (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count() : std::strtoull(seed.c_str(), nullptr, 10));
    int len = std::max(1, std::atoi(flag(a, "--len", "512").c_str()));
    auto timed = [&](const std::vector<int>& ids, const Sampling& s) {
        auto t = std::chrono::steady_clock::now();
        m->reply(ids, len, s, rng, [&](const std::string& part) {
            std::fputs(part.c_str(), stdout);
            std::fflush(stdout);
            return true;
        });
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        std::printf("\n[%zu prompt tokens, %.1f s]\n", ids.size(), sec);
    };
    if (sub == "generate") {
        std::string prompt = a.size() > 4 && a[4].compare(0, 2, "--") != 0 ? a[4] : "";
        timed(m->encode(prompt, true, false), sampling_flags(a, "0.8", "0.95", "40", "1.1"));
        return 0;
    }
    if (sub == "chat") {
        Sampling s = sampling_flags(a, "0.7", "0.9", "40", "1.1");
        std::vector<LlmMsg> msgs;
        std::string sys = flag(a, "--system");
        if (!sys.empty()) msgs.push_back({"system", sys});
        std::string once = flag(a, "--prompt");
        std::string line;
        while (true) {
            if (once.empty()) {
                std::printf("\n> ");
                std::fflush(stdout);
                if (!read_line(line) || line == "/exit" || line == "/quit") break;
                if (line.empty()) continue;
                if (line == "/clear") {
                    msgs.resize(sys.empty() ? 0 : 1);
                    continue;
                }
            } else {
                line = once;
            }
            msgs.push_back({"user", line});
            auto t = std::chrono::steady_clock::now();
            std::string answer = m->reply(m->chat_prompt(msgs, len), len, s, rng, [&](const std::string& part) {
                std::fputs(part.c_str(), stdout);
                std::fflush(stdout);
                return true;
            });
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
            size_t n = m->encode(answer, false, false).size();
            std::printf("\n[%zu tokens, %.1f s, %.1f tokens/s]\n", n, sec, n / std::max(1e-6, sec));
            msgs.push_back({"assistant", answer});
            if (!once.empty()) break;
        }
        return 0;
    }
    if (sub == "lora" && a.size() > 4 && a[4].compare(0, 2, "--") != 0) {
        std::string text;
        if (!read_text(a[4], text)) {
            std::printf("error: cannot read %s\n", a[4].c_str());
            return 1;
        }
        LoraConfig lc;
        lc.rank = std::max(1, std::atoi(flag(a, "--rank", "8").c_str()));
        lc.alpha = (float)std::atof(flag(a, "--alpha", std::to_string(2 * lc.rank).c_str()).c_str());
        lc.mlp = has_flag(a, "--mlp");
        lc.steps = std::max(1, std::atoi(flag(a, "--steps", "200").c_str()));
        lc.batch = std::max(1, std::atoi(flag(a, "--batch", "4").c_str()));
        lc.seq = std::max(8, std::atoi(flag(a, "--seq", "256").c_str()));
        lc.lr = (float)std::atof(flag(a, "--lr", "2e-4").c_str());
        lc.eval_every = std::max(1, std::atoi(flag(a, "--eval-every", "25").c_str()));
        std::string out = flag(a, "--out", (upath(path).replace_extension("").u8string() + "-lora.gguf").c_str());
        LoraData d = lora_data(*m, text, flag(a, "--q-tag", "Kullanıcı: "), flag(a, "--a-tag", "Asistan: "), flag(a, "--system"));
        std::printf("training data: %zu %s, %zu tokens\n", d.ex.size(), d.chat ? "question-answer pairs (only answers are learned)" : "text",
                    d.tokens());
        static std::atomic<bool> stop{false};
        std::signal(SIGINT, [](int) { stop = true; });
        bool ok = lora_train(*m, d, lc, [&](const LoraProgress& p) {
            char val[48] = "";
            if (p.val >= 0) std::snprintf(val, sizeof val, "  validation %.4f", p.val);
            std::printf("step %d/%d  loss %.4f%s  %.0f tokens/s\n", p.step, p.steps, p.loss, val, p.tok_s);
            std::fflush(stdout);
            return !stop.load();
        }, &err);
        if (!ok) {
            std::printf("error: %s\n", err.c_str());
            return 1;
        }
        if (!m->lora_save(out, &err)) {
            std::printf("error: %s\n", err.c_str());
            return 1;
        }
        std::printf("saved the LoRA adapter to %s (use it with --lora, or in llama.cpp with --lora)\n", out.c_str());
        return 0;
    }
    std::printf("unknown gguf command\n");
    return 1;
}

int main(int argc, char** argv) {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::vector<std::string> a = utf8_args(argc, argv);
    std::string cmd = a.size() > 1 ? a[1] : "";
    if (cmd == "sysinfo") { print_sysinfo(); return 0; }
    if (cmd == "selftest") return selftest(has_flag(a, "--quick"));
    if (cmd == "gpu") return gpu_check(has_flag(a, "--quick"));
    if (cmd == "demo-data" && a.size() > 2) {
        bool ok = write_spiral_csv(a[2]);
        std::printf(ok ? "wrote %s\n" : "cannot write %s\n", a[2].c_str());
        return ok ? 0 : 1;
    }
    if ((cmd == "demo-images" || cmd == "demo-sounds") && a.size() > 2) {
        std::string err;
        bool ok = cmd == "demo-images" ? write_demo_images(a[2], 1000, &err) : write_demo_sounds(a[2], 60, &err);
        if (ok) std::printf("wrote %s\n", a[2].c_str());
        else std::printf("error: %s: %s\n", a[2].c_str(), err.c_str());
        return ok ? 0 : 1;
    }
    if (cmd == "train" && a.size() > 2) return train(a);
    if (cmd == "generate" && a.size() > 2) return generate(a);
    if (cmd == "test" && a.size() > 3) return test(a);
    if (cmd == "serve" && a.size() > 2) return serve(a);
    if (cmd == "data" && a.size() > 3) return data_tool(a);
    if (cmd == "gguf" && a.size() > 3) return gguf_cmd(a);
    std::printf("Kor Studio CLI\n"
                "  kor-cli sysinfo                      show detected hardware\n"
                "  kor-cli selftest [--quick]           verify the training engine (--quick: no training runs)\n"
                "  kor-cli gpu                          check the OpenCL GPU against the CPU (experimental)\n"
                "  kor-cli demo-data <out.csv>          write a demo classification dataset\n"
                "  kor-cli demo-images <folder>         write demo pictures (for an image model)\n"
                "  kor-cli demo-sounds <folder>         write demo melodies (for a sound model)\n"
                "  kor-cli train <data> [--steps N] [--patience N] [--tests tests.txt] [--target 0..1]\n"
                "                [--log train.log] [--out model.kor] [--text|--table|--images|--audio]\n"
                "                [--subword N]                        text: N subword tokens instead of characters\n"
                "                [--size 16] [--colors 32] [--no-mirror]   images (a folder of PNG/BMP/PPM)\n"
                "                [--rate 8000]                        sound (a WAV file or a folder of them)\n"
                "                [--skill \"skill.py [args]\"]... [--python <command>] [--lang en|tr|ru]\n"
                "                [--from model.kor [--lora R]]        fine-tune a saved model\n"
                "                [--name NAME] [--maker WHO] [--no-teach-name]   the model's name; with dialogue\n"
                "                      data it also learns to answer \"What is your name?\" with it\n"
                "                [--dim N] [--layers N] [--heads N] [--ctx N]   text model size\n"
                "                [--arch modern|classic] [--ffn N]   modern (text default): RMSNorm, RoPE, SwiGLU\n"
                "                [--recompute]                        less memory: blocks run again in the backward pass\n"
                "                [--lr X] [--batch N] [--threads N] [--seed N] [--wd X] [--gpu]\n"
                "                [--align 0..1] [--prompt-weight 0..1] [--dropout X] [--ema 0.999]\n"
                "                [--resume|--fresh] [--checkpoint file] [--checkpoint-every 60] [--no-checkpoint]\n"
                "                      autosave (default <out>.autosave); Ctrl+C stops and saves\n"
                "  kor-cli generate <model.kor> [prompt] [--len N] [--temp T] [--line] [--seed N]\n"
                "                [--top-p 0.9] [--top-k 40] [--repeat 1.1]   sampling\n"
                "                [--gpu]                              generate on the graphics card\n"
                "                image model: [--count 16] [--out generated.png]\n"
                "                sound model: [--seconds 3] [--out generated.wav]\n"
                "  kor-cli test <model.kor> <tests.txt> [--data <text>] [--tsv]\n"
                "  kor-cli serve <model.kor> [--port 8080] [--host 127.0.0.1] [--api-key KEY] [--cors]\n"
                "                [--name ID] [--q-tag \"Kullanıcı: \"] [--a-tag \"Asistan: \"] [--gpu]\n"
                "                      OpenAI-compatible API: /v1/chat/completions, /v1/completions, /v1/models\n"
                "  kor-cli data clean <text> [--out file] [--keep-duplicates] [--keep-empty]\n"
                "                      tidy spaces, drop repeated and broken question-answer pairs\n"
                "  kor-cli data import <file.jsonl> [--out file] [--q-tag \"Kullanıcı: \"] [--a-tag \"Asistan: \"]\n"
                "                      chat datasets (messages, ShareGPT, Alpaca, prompt/completion) to training text\n"
                "  kor-cli gguf info|chat|generate <model.gguf> [prompt]   ready-made Llama / Mistral / Qwen2 models\n"
                "                [--prompt \"...\"] [--system \"...\"] [--lora adapter.gguf] [--ctx 4096] [--len N]\n"
                "                [--temp T] [--top-p P] [--top-k K] [--repeat R] [--threads N] [--seed N]\n"
                "  kor-cli gguf lora <model.gguf> <data.txt> [--out adapter.gguf] [--rank 8] [--alpha 16] [--mlp]\n"
                "                [--steps 200] [--batch 4] [--seq 256] [--lr 2e-4] [--q-tag ...] [--a-tag ...]\n"
                "                      LoRA fine-tuning on the frozen model; dialogue data trains the answers only\n"
                "\n"
                "Question tests are written one per line as: question => expected answer\n");
    return cmd.empty() ? 0 : 1;
}
