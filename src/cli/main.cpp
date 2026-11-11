// kor-cli: headless companion to Kor Studio (system info, training, self-test,
// generation and question tests). Also used by Python plugins.
#include "../core/kernels.h"
#include "../core/media.h"
#include "../core/parallel.h"
#include "../core/sysinfo.h"
#include "../skills/skill.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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

static int selftest() {
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
    {   // transformer
        ModelSpec s;
        s.kind = MODEL_GPT; s.vocab = 11; s.ctx = 6; s.dim = 16; s.heads = 2; s.nlayer = 2;
        Model m(s);
        for (Param* p : m.params()) for (auto& v : p->w.v) v += rng.normal() * 0.15f;  // leave the flat init region
        Batch b;
        b.B = 3; b.T = 6;
        for (int i = 0; i < 18; ++i) { b.ids.push_back(rng.below(11)); b.tgt.push_back(rng.below(11)); }
        if (gradcheck(m, b, "transformer") >= 0.08) ++fails;

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
        if (gradcheck(m, b, "gpt+lora") >= 0.08) ++fails;
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
        auto v = learn_subwords(chars, 40);
        std::vector<int> ids = tokenize(text, v);
        std::string back;
        for (int t : ids) back += v[t];
        bool no_nl_merge = true;
        for (auto& p : v) no_nl_merge &= p == "\n" || p.find('\n') == std::string::npos;
        bool pass = back == text && ids.size() < chars.size() / 2 && no_nl_merge;
        std::printf("subword tokens: %zu pieces, %zu chars -> %zu tokens %s\n", v.size(), chars.size(), ids.size(), pass ? "OK" : "FAIL");
        if (!pass) ++fails;
    }
    {   // crash-safe training: a run stopped halfway and resumed ends like an uninterrupted one
        std::string text;
        for (int i = 0; i < 120; ++i) text += "Kullanıcı: merhaba " + std::to_string(i % 7) + "\nAsistan: selam " + std::to_string(i % 5) + "\n";
        std::string err, dpath = "kor_selftest_data.txt", ck = "kor_selftest_ckpt.kor";
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
            void on_event(const HookInfo&, TrainControl&) override { tr->request_stop(); }
        };
        auto run = [&](bool interrupt, std::vector<float>& w_out) {
            ModelSpec s;
            s.kind = MODEL_GPT; s.vocab = (int)data.vocab.size(); s.ctx = 16; s.dim = 16; s.heads = 2; s.nlayer = 1;
            auto m = std::make_shared<Model>(s);
            m->vocab = data.vocab;
            TrainConfig c;
            c.steps = 40; c.batch = 4; c.eval_every = 10; c.ema = 0.9f; c.threads = 1;
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
        pass = pass && diff < 1e-4f;
        remove_checkpoint(ck);
        std::printf("resume after stop: weight difference %.2g %s %s\n", diff, pass ? "OK" : "FAIL", err.c_str());
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
    AutoPlan plan = autotune(sys, data);
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
    std::printf("auto plan: %zu parameters, ~%.0f MB RAM, batch %d x%d, %d steps, %d threads, kernels %s\n",
                plan.params, plan.est_ram_mb, plan.cfg.batch, plan.cfg.accum, plan.cfg.steps, plan.cfg.threads, K().name);
    if (kind == DATA_TEXT)
        std::printf("model: dim %d, %d layers, %d heads, context %d; lr %g; aligned rows %.0f%%, question weight %g, dropout %g, weight decay %g, ema %g\n",
                    plan.spec.dim, plan.spec.nlayer, plan.spec.heads, plan.spec.ctx, plan.cfg.lr, plan.cfg.align * 100,
                    plan.cfg.prompt_weight, plan.cfg.dropout, plan.cfg.weight_decay, plan.cfg.ema);
    if (media == MEDIA_IMAGE) std::printf("image model: %dx%d pixels, %zu colours\n", data.img_size, data.img_size, data.vocab.size() - 1);
    if (media == MEDIA_AUDIO) std::printf("sound model: %d Hz, 256 levels (mu-law)\n", data.rate);
    auto model = base ? base : std::make_shared<Model>(plan.spec);
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
    for (int t : m->generate(ids, len, temp, rng, -1, [&](int t) { return !(line && m->vocab[t] == "\n"); })) {
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

int main(int argc, char** argv) {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::vector<std::string> a = utf8_args(argc, argv);
    std::string cmd = a.size() > 1 ? a[1] : "";
    if (cmd == "sysinfo") { print_sysinfo(); return 0; }
    if (cmd == "selftest") return selftest();
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
    std::printf("Kor Studio CLI\n"
                "  kor-cli sysinfo                      show detected hardware\n"
                "  kor-cli selftest                     verify the training engine\n"
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
                "                [--dim N] [--layers N] [--heads N] [--ctx N]   text model size\n"
                "                [--lr X] [--batch N] [--threads N] [--seed N] [--wd X]\n"
                "                [--align 0..1] [--prompt-weight 0..1] [--dropout X] [--ema 0.999]\n"
                "                [--resume|--fresh] [--checkpoint file] [--checkpoint-every 60] [--no-checkpoint]\n"
                "                      autosave (default <out>.autosave); Ctrl+C stops and saves\n"
                "  kor-cli generate <model.kor> [prompt] [--len N] [--temp T] [--line] [--seed N]\n"
                "                image model: [--count 16] [--out generated.png]\n"
                "                sound model: [--seconds 3] [--out generated.wav]\n"
                "  kor-cli test <model.kor> <tests.txt> [--data <text>] [--tsv]\n"
                "\n"
                "Question tests are written one per line as: question => expected answer\n");
    return cmd.empty() ? 0 : 1;
}
