// kor-cli: headless companion to Kor Studio (system info, training, self-test).
#include "../core/kernels.h"
#include "../core/parallel.h"
#include "../core/sysinfo.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

using namespace kor;

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

static int train(int argc, char** argv) {
    std::string path = argv[2], out;
    int steps = 0, kind = 0;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--steps") && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!std::strcmp(argv[i], "--table")) kind = DATA_TABLE;
        else if (!std::strcmp(argv[i], "--text")) kind = DATA_TEXT;
    }
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
    Trainer tr;
    if (!tr.start(model, &data, plan.cfg)) { std::printf("error: %s\n", tr.stats().error.c_str()); return 1; }
    size_t shown = 0;
    while (tr.running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        TrainStats st = tr.stats();
        for (; shown < st.val_hist.size(); ++shown)
            std::printf("step %5d  val loss %.4f\n", st.val_steps[shown], st.val_hist[shown]);
        std::fflush(stdout);
    }
    TrainStats st = tr.stats();
    for (; shown < st.val_hist.size(); ++shown)
        std::printf("step %5d  val loss %.4f\n", st.val_steps[shown], st.val_hist[shown]);
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

int main(int argc, char** argv) {
    std::string cmd = argc > 1 ? argv[1] : "";
    if (cmd == "sysinfo") { print_sysinfo(); return 0; }
    if (cmd == "selftest") return selftest();
    if (cmd == "demo-data" && argc > 2) {
        bool ok = write_spiral_csv(argv[2]);
        std::printf(ok ? "wrote %s\n" : "cannot write %s\n", argv[2]);
        return ok ? 0 : 1;
    }
    if (cmd == "train" && argc > 2) return train(argc, argv);
    std::printf("Kor Studio CLI\n"
                "  kor-cli sysinfo                      show detected hardware\n"
                "  kor-cli selftest                     verify the training engine\n"
                "  kor-cli demo-data <out.csv>          write a demo classification dataset\n"
                "  kor-cli train <data> [--steps N] [--out model.kor] [--text|--table]\n");
    return cmd.empty() ? 0 : 1;
}
