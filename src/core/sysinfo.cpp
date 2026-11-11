#include "sysinfo.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define KOR_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
static void cpuid_(int leaf, int sub, unsigned r[4]) {
    int t[4];
    __cpuidex(t, leaf, sub);
    for (int i = 0; i < 4; ++i) r[i] = (unsigned)t[i];
}
static unsigned long long xgetbv_() { return _xgetbv(0); }
#else
#include <cpuid.h>
static void cpuid_(int leaf, int sub, unsigned r[4]) { __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]); }
static unsigned long long xgetbv_() {
    unsigned a, d;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(a), "=d"(d) : "c"(0));
    return ((unsigned long long)d << 32) | a;
}
#endif
#endif

namespace kor {

#ifdef KOR_X86
namespace {
struct Features { bool avx2 = false, avx512 = false; std::string brand; };
const Features& features() {
    static const Features f = [] {
        Features x;
        unsigned r[4];
        cpuid_(0, 0, r);
        unsigned max_leaf = r[0];
        cpuid_(1, 0, r);
        bool osxsave = (r[2] >> 27) & 1, fma = (r[2] >> 12) & 1;
        unsigned long long xcr = osxsave ? xgetbv_() : 0;
        bool ymm = (xcr & 0x6) == 0x6, zmm = (xcr & 0xE6) == 0xE6;
        if (max_leaf >= 7) {
            cpuid_(7, 0, r);
            x.avx2 = ymm && fma && ((r[1] >> 5) & 1);
            x.avx512 = zmm && ((r[1] >> 16) & 1);
        }
        cpuid_(0x80000000, 0, r);
        if (r[0] >= 0x80000004u) {
            char buf[49] = {0};
            for (int i = 0; i < 3; ++i) {
                cpuid_(0x80000002 + i, 0, r);
                std::memcpy(buf + i * 16, r, 16);
            }
            x.brand = buf;
            size_t a = x.brand.find_first_not_of(' '), b = x.brand.find_last_not_of(' ');
            x.brand = a == std::string::npos ? "" : x.brand.substr(a, b - a + 1);
        }
        return x;
    }();
    return f;
}
}  // namespace
bool cpu_has_avx2() { return features().avx2; }
#else
bool cpu_has_avx2() { return false; }
#endif

SysInfo detect_system() {
    SysInfo s;
    s.logical = std::max(1u, std::thread::hardware_concurrency());
#ifdef KOR_X86
    if (!features().brand.empty()) s.cpu = features().brand;
    s.avx2 = features().avx2;
    s.avx512 = features().avx512;
#endif
    const double GB = 1024.0 * 1024.0 * 1024.0;
#if defined(_WIN32)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        s.ram_total_gb = (double)ms.ullTotalPhys / GB;
        s.ram_free_gb = (double)ms.ullAvailPhys / GB;
    }
    DISPLAY_DEVICEA dd;
    for (DWORD i = 0;; ++i) {
        std::memset(&dd, 0, sizeof(dd));
        dd.cb = sizeof(dd);
        if (!EnumDisplayDevicesA(nullptr, i, &dd, 0)) break;
        std::string name = dd.DeviceString;
        if (!name.empty() && std::find(s.gpus.begin(), s.gpus.end(), name) == s.gpus.end()) s.gpus.push_back(name);
    }
#elif defined(__APPLE__)
    uint64_t mem = 0;
    size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) s.ram_total_gb = (double)mem / GB;
    char brand[128];
    len = sizeof(brand);
    if (sysctlbyname("machdep.cpu.brand_string", brand, &len, nullptr, 0) == 0) s.cpu = brand;
    vm_statistics64_data_t vm;
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &cnt) == KERN_SUCCESS)
        s.ram_free_gb = (double)(vm.free_count + vm.inactive_count) * (double)sysconf(_SC_PAGESIZE) / GB;
    else
        s.ram_free_gb = s.ram_total_gb * 0.5;
#else
    std::ifstream mi("/proc/meminfo");
    std::string key;
    double kb;
    std::string unit;
    while (mi >> key >> kb >> unit) {
        if (key == "MemTotal:") s.ram_total_gb = kb * 1024.0 / GB;
        else if (key == "MemAvailable:") s.ram_free_gb = kb * 1024.0 / GB;
    }
    if (s.ram_total_gb <= 0) {
        s.ram_total_gb = (double)sysconf(_SC_PHYS_PAGES) * (double)sysconf(_SC_PAGESIZE) / GB;
        s.ram_free_gb = s.ram_total_gb * 0.5;
    }
#endif
    return s;
}

int recommended_threads(const SysInfo& s) {
    // Logical cores are usually 2 per physical core; matrix math gains little
    // from hyper-threads, and we leave room for the UI and the OS.
    if (s.logical >= 8) return s.logical / 2;
    return std::max(1, s.logical - 1);
}

static int pow2_at_least(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

double estimate_gpu_train_bytes(const ModelSpec& s) {
    // as gpt_act_floats in gpu.cpp for one sequence of s.ctx tokens, training
    double T = s.ctx, C = s.dim, att = (double)s.heads * T * T;
    double act = s.nlayer * T * (12 * C + 4) + 7 * T * C + T * s.vocab + 11 * T + 2 * att + 13 * T * C;
    return ((double)estimate_params(s) * 4 + act) * 4;
}

AutoPlan autotune(const SysInfo& s, const Dataset& d, double gpu_gb) {
    AutoPlan p;
    int threads = recommended_threads(s);
    p.cfg.threads = threads;
    double budget = std::min(s.ram_free_gb * 0.5, s.ram_total_gb * 0.4) * 1024.0;  // MB
    if (budget < 128) budget = 128;
    p.budget_mb = budget;
    double power = threads * (s.avx2 ? 2.0 : 1.0);

    if (d.kind == DATA_TEXT) {
        struct Preset { int dim, heads, layers, ctx; };
        // 0-4 for the CPU; 5-9 (about 14M, 40M, 85M, 200M, 300M parameters) only on a GPU
        static const Preset presets[10] = {
            {48, 2, 2, 48},     {64, 4, 3, 64},      {128, 4, 4, 96},     {192, 6, 6, 128},    {256, 8, 6, 192},
            {384, 6, 8, 256},   {512, 8, 12, 512},   {768, 12, 12, 512},  {1024, 16, 16, 512}, {1024, 16, 24, 1024}};
        const bool on_gpu = gpu_gb > 0 && d.media == MEDIA_TEXT;
        int by_power = power < 4 ? 0 : power < 8 ? 1 : power < 16 ? 2 : power < 32 ? 3 : 4;
        if (on_gpu) by_power = 9;
        size_t n = d.ids.size();
        // A model learns from text in proportion to its size: large ones need
        // tens to hundreds of millions of tokens, or they only memorize.
        int by_data = n < 50000 ? 0 : n < 300000 ? 1 : n < 2000000 ? 2 : n < 20000000 ? 3 : n < 60000000 ? 4
                    : n < 200000000 ? 5 : n < 500000000 ? 6 : n < 1000000000 ? 7 : 8;
        if (!on_gpu) by_data = std::min(by_data, 4);
        std::string q_tag, a_tag;
        bool dialogue = d.media == MEDIA_TEXT && d.dialogue_tags(q_tag, a_tag);
        // Exact answers need more model than the amount of text suggests: on
        // small dialogue sets the nano model stays at a few correct answers.
        if (dialogue) by_data = std::max(by_data, 1);
        // Images: a few hundred pictures are memorized by a 1M-parameter model.
        if (d.media == MEDIA_IMAGE) by_data = d.items < 1000 ? 1 : d.items < 10000 ? 2 : 3;
        if (d.media == MEDIA_AUDIO) by_data = std::max(by_data, 1);
        int tier = std::min(by_power, by_data);
        // Context: a whole image, a stretch of sound, or (dialogue) most
        // question-answer pairs in one window.
        int want_ctx = 0;
        if (d.media == MEDIA_IMAGE) want_ctx = d.img_size * d.img_size + 1;
        else if (d.media == MEDIA_AUDIO) want_ctx = tier <= 1 ? 256 : 512;
        else if (dialogue && d.layout.starts.size() > 2) {
            std::vector<int> len;
            for (size_t i = 0; i + 1 < d.layout.starts.size(); ++i)
                len.push_back((int)(d.layout.starts[i + 1] - d.layout.starts[i]));
            std::nth_element(len.begin(), len.begin() + len.size() * 9 / 10, len.end());
            int p90 = len[len.size() * 9 / 10];
            want_ctx = std::min(256, (p90 * 5 / 4 + 15) / 16 * 16);
        }
        int batch = 32;
        ModelSpec spec;
        for (;;) {
            const Preset& pr = presets[tier];
            spec = ModelSpec();
            spec.kind = MODEL_GPT;
            spec.vocab = (int)d.vocab.size();
            spec.dim = pr.dim; spec.heads = pr.heads; spec.nlayer = pr.layers;
            spec.media = d.media;
            spec.img_size = d.img_size;
            spec.rate = d.rate;
            if (d.media == MEDIA_IMAGE) spec.ctx = want_ctx;
            else spec.ctx = std::max(8, std::min(std::max(pr.ctx, want_ctx), (int)d.n_train / 4));
            if (tier >= 5) {
                // On the card: the model, its optimizer state and one sequence must
                // fit (larger batches run in parts); the host keeps weights,
                // gradients, moments and a best-weights copy.
                double gpu_mb = estimate_gpu_train_bytes(spec) / (1024.0 * 1024.0);
                double host_mb = (double)estimate_params(spec) * 4 * 5 / (1024.0 * 1024.0);
                if (gpu_mb <= gpu_gb * 1024.0 * 0.8 && host_mb <= budget) {
                    batch = spec.ctx >= 1024 ? 8 : 16;
                    p.est_gpu_mb = gpu_mb;
                    break;
                }
                --tier;
                continue;
            }
            double mb = (double)estimate_train_bytes(spec, batch) / (1024.0 * 1024.0);
            if (mb <= budget) break;
            if (batch > 4) batch /= 2;
            else if (tier > 0) { --tier; batch = 32; }
            else break;
        }
        p.tier = tier;
        p.spec = spec;
        p.cfg.batch = batch;
        p.cfg.accum = batch < 16 ? 16 / batch : 1;
        if (tier >= 5) p.cfg.accum = std::max(1, 64 / batch);  // about 32k tokens per step
        // larger models need smaller steps
        static const float lrs[10] = {3e-3f, 3e-3f, 1e-3f, 1e-3f, 1e-3f, 6e-4f, 4e-4f, 3e-4f, 2e-4f, 2e-4f};
        p.cfg.lr = lrs[tier];
        double per_step = (double)batch * p.cfg.accum * spec.ctx;
        // Dialogue has to be learned answer by answer, which takes far more
        // passes than picking up the style of ordinary text. Question tests
        // can stop such a run early once the answers are right. Images and
        // sound are learned from their own pixels and samples, many passes.
        double epochs = dialogue ? 250.0 : d.media == MEDIA_IMAGE ? 60.0 : d.media == MEDIA_AUDIO ? 20.0 : 4.0;
        bool long_run = dialogue || d.media != MEDIA_TEXT;
        // a large model on a large text: about one pass over it
        if (tier >= 5) epochs = 1.0;
        double most = tier >= 5 ? 100000.0 : long_run ? 8000.0 : 3000.0;
        p.cfg.steps = (int)std::max(long_run ? 1000.0 : 300.0, std::min(most, (double)d.n_train * epochs / per_step));
        p.cfg.warmup = std::min(tier >= 5 ? 2000 : 100, p.cfg.steps / 10);
        if (dialogue) {
            // Windows that start at a question and a lower weight on the
            // question text: the loss goes where the answers are.
            p.cfg.align = 0.75f;
            p.cfg.prompt_weight = 0.2f;
        } else if (d.media == MEDIA_IMAGE) {
            p.cfg.align = 1.0f;  // a window is exactly one image
            if (d.items < 3000) p.cfg.dropout = 0.1f;
        } else if (d.media == MEDIA_AUDIO) {
            p.cfg.align = 0.25f;
        }
    } else if (d.kind == DATA_TABLE) {
        int n = (int)d.train_idx.size();
        int width = std::max(16, std::min(256, pow2_at_least(d.n_feat * 4)));
        if (n < 500) width = std::min(width, 32);
        else if (n < 5000) width = std::min(width, 64);
        width = std::max(width, 16);
        ModelSpec spec;
        spec.kind = MODEL_MLP;
        spec.in_dim = d.n_feat;
        spec.regression = d.regression;
        spec.out_dim = d.regression ? 1 : (int)d.labels.size();
        LayerSpec dense; dense.type = L_DENSE; dense.units = width;
        LayerSpec relu; relu.type = L_RELU;
        spec.layers = {dense, relu, dense, relu};
        p.spec = spec;
        int batch = std::max(8, std::min(128, pow2_at_least(std::max(1, n / 24))));
        p.cfg.batch = batch;
        p.cfg.accum = 1;
        p.cfg.lr = 3e-3f;
        p.cfg.steps = (int)std::max(300.0, std::min(4000.0, (double)n * 60.0 / batch));
        p.cfg.warmup = 20;
        p.tier = width <= 32 ? 0 : width <= 64 ? 1 : 2;
    }
    p.cfg.eval_every = std::max(10, p.cfg.steps / 40);
    p.params = estimate_params(p.spec);
    p.est_ram_mb = (double)estimate_train_bytes(p.spec, p.cfg.batch) / (1024.0 * 1024.0);
    return p;
}

}  // namespace kor
