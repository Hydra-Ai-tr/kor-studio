// kor-cli: headless companion to Kor Studio (system info, training, self-test,
// generation and question tests). Also used by Python plugins.
#include "../core/kernels.h"
#include "../core/parallel.h"
#include "../core/sysinfo.h"
#include "../skills/skill.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
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
    if (!kind) {
        std::string ext = path.size() > 4 ? path.substr(path.size() - 4) : "";
        kind = (ext == ".csv" || ext == ".tsv") ? DATA_TABLE : DATA_TEXT;
    }
    Dataset data;
    std::string err;
    bool ok = kind == DATA_TABLE ? data.load_table(path, -1, &err) : data.load_text(path, &err);
    if (!ok) { std::printf("error: %s\n", err.c_str()); return 1; }
    SysInfo sys = detect_system();
    AutoPlan plan = autotune(sys, data);
    if (steps > 0) { plan.cfg.steps = steps; plan.cfg.eval_every = std::max(10, steps / 10); }
    else plan.cfg.eval_every = std::max(10, plan.cfg.steps / 10);
    std::printf("auto plan: %zu parameters, ~%.0f MB RAM, batch %d x%d, %d steps, %d threads, kernels %s\n",
                plan.params, plan.est_ram_mb, plan.cfg.batch, plan.cfg.accum, plan.cfg.steps, plan.cfg.threads, K().name);
    auto model = std::make_shared<Model>(plan.spec);
    if (kind == DATA_TEXT) model->vocab = data.vocab;
    else { model->vocab = data.labels; model->norm_mean = data.mean; model->norm_std = data.stdv; }
    if (kind == DATA_TEXT) {
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
            }
        }
        std::fflush(stdout);
        if (log) log.flush();
    };
    while (tr.running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        print_events();
    }
    print_events();
    TrainStats st = tr.stats();
    if (st.state == TS_ERROR) { std::printf("error: %s\n", st.error.c_str()); return 1; }
    std::printf("done in %.1fs: train loss %.4f, val loss %.4f", st.elapsed, st.loss, st.val_loss);
    if (st.val_acc >= 0) std::printf(", val accuracy %.1f%%", st.val_acc * 100);
    std::printf(", %.0f %s/s\n", st.items_per_sec, kind == DATA_TEXT ? "tokens" : "samples");
    if (kind == DATA_TEXT) {
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
    float temp = (float)std::atof(flag(a, "--temp", "0.8").c_str());
    bool line = has_flag(a, "--line");
    Rng rng((uint64_t)std::chrono::steady_clock::now().time_since_epoch().count());
    std::vector<int> ids = m->encode(prompt);
    if (ids.empty()) ids.push_back(rng.below(std::max(1, m->spec.vocab)));
    std::string out;
    for (int i = 0; i < len; ++i) {
        int t = m->next_token(ids, temp, rng);
        if (line && m->vocab[t] == "\n") break;
        ids.push_back(t);
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
    if (cmd == "train" && a.size() > 2) return train(a);
    if (cmd == "generate" && a.size() > 2) return generate(a);
    if (cmd == "test" && a.size() > 3) return test(a);
    std::printf("Kor Studio CLI\n"
                "  kor-cli sysinfo                      show detected hardware\n"
                "  kor-cli selftest                     verify the training engine\n"
                "  kor-cli demo-data <out.csv>          write a demo classification dataset\n"
                "  kor-cli train <data> [--steps N] [--patience N] [--tests tests.txt] [--target 0..1]\n"
                "                [--log train.log] [--out model.kor] [--text|--table]\n"
                "                [--skill \"skill.py [args]\"]... [--python <command>] [--lang en|tr|ru]\n"
                "  kor-cli generate <model.kor> [prompt] [--len N] [--temp T] [--line]\n"
                "  kor-cli test <model.kor> <tests.txt> [--data <text>] [--tsv]\n"
                "\n"
                "Question tests are written one per line as: question => expected answer\n");
    return cmd.empty() ? 0 : 1;
}
