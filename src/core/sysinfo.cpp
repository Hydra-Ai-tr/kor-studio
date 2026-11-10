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

AutoPlan autotune(const SysInfo& s, const Dataset& d) {
    AutoPlan p;
    int threads = recommended_threads(s);
    p.cfg.threads = threads;
    double budget = std::min(s.ram_free_gb * 0.5, s.ram_total_gb * 0.4) * 1024.0;  // MB
    if (budget < 128) budget = 128;
    p.budget_mb = budget;
    double power = threads * (s.avx2 ? 2.0 : 1.0);

    if (d.kind == DATA_TEXT) {
        struct Preset { int dim, heads, layers, ctx; };
        static const Preset presets[5] = {
            {48, 2, 2, 48}, {64, 4, 3, 64}, {128, 4, 4, 96}, {192, 6, 6, 128}, {256, 8, 6, 192}};
        int by_power = power < 4 ? 0 : power < 8 ? 1 : power < 16 ? 2 : power < 32 ? 3 : 4;
        size_t n = d.ids.size();
        int by_data = n < 50000 ? 0 : n < 300000 ? 1 : n < 2000000 ? 2 : n < 20000000 ? 3 : 4;
        int tier = std::min(by_power, by_data);
        int batch = 32;
        ModelSpec spec;
        for (;;) {
            const Preset& pr = presets[tier];
            spec = ModelSpec();
            spec.kind = MODEL_GPT;
            spec.vocab = (int)d.vocab.size();
            spec.dim = pr.dim; spec.heads = pr.heads; spec.nlayer = pr.layers;
            spec.ctx = std::max(8, std::min(pr.ctx, (int)d.n_train / 4));
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
        p.cfg.lr = tier <= 1 ? 3e-3f : 1e-3f;
        double per_step = (double)batch * p.cfg.accum * spec.ctx;
        // Dialogue has to be learned answer by answer, which takes far more
        // passes than picking up the style of ordinary text. Question tests
        // can stop such a run early once the answers are right.
        std::string q_tag, a_tag;
        bool dialogue = d.dialogue_tags(q_tag, a_tag);
        double epochs = dialogue ? 250.0 : 4.0;
        p.cfg.steps = (int)std::max(dialogue ? 1000.0 : 300.0, std::min(dialogue ? 8000.0 : 3000.0, (double)d.n_train * epochs / per_step));
        p.cfg.warmup = std::min(100, p.cfg.steps / 10);
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
