#include "gpu.h"

#include "kernels.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define CLAPI __stdcall
#else
#include <dlfcn.h>
#define CLAPI
#endif

namespace kor {
namespace {

// The few OpenCL 1.2 declarations used here, so no SDK is needed to build.
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_mem;
typedef void* cl_event;

const cl_ulong CL_DEVICE_TYPE_GPU = 1 << 2;
const cl_uint CL_PLATFORM_NAME = 0x0902;
const cl_uint CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004;
const cl_uint CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F;
const cl_uint CL_DEVICE_NAME = 0x102B;
const cl_uint CL_DEVICE_VENDOR = 0x102C;
const cl_uint CL_PROGRAM_BUILD_LOG = 0x1183;
const cl_ulong CL_MEM_READ_WRITE = 1;

struct Cl {
    cl_int(CLAPI* GetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
    cl_int(CLAPI* GetPlatformInfo)(cl_platform_id, cl_uint, size_t, void*, size_t*);
    cl_int(CLAPI* GetDeviceIDs)(cl_platform_id, cl_ulong, cl_uint, cl_device_id*, cl_uint*);
    cl_int(CLAPI* GetDeviceInfo)(cl_device_id, cl_uint, size_t, void*, size_t*);
    cl_context(CLAPI* CreateContext)(const intptr_t*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
    cl_command_queue(CLAPI* CreateCommandQueue)(cl_context, cl_device_id, cl_ulong, cl_int*);
    cl_program(CLAPI* CreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
    cl_int(CLAPI* BuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
    cl_int(CLAPI* GetProgramBuildInfo)(cl_program, cl_device_id, cl_uint, size_t, void*, size_t*);
    cl_kernel(CLAPI* CreateKernel)(cl_program, const char*, cl_int*);
    cl_mem(CLAPI* CreateBuffer)(cl_context, cl_ulong, size_t, void*, cl_int*);
    cl_int(CLAPI* ReleaseMemObject)(cl_mem);
    cl_int(CLAPI* EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
    cl_int(CLAPI* EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
    cl_int(CLAPI* SetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
    cl_int(CLAPI* EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint,
                                        const cl_event*, cl_event*);
    cl_int(CLAPI* Finish)(cl_command_queue);
};

// C[M x N] += A'[M x P] * B'[P x N], with A'(i, p) = A[i * sai + p * sap] and
// B'(p, j) = B[p * sbp + j * sbj]: the three products of Kernels in one kernel.
// A 32 x 32 tile per work-group of 32 x 8; every work-item sums 4 rows.
const char* kSource = R"CL(
#define TS 32
#define WPT 4
#define RTS 8
__kernel __attribute__((reqd_work_group_size(TS, RTS, 1)))
void gemm(const int M, const int N, const int P,
          __global const float* A, const int sai, const int sap,
          __global const float* B, const int sbp, const int sbj,
          __global float* C) {
    const int lj = get_local_id(0), li = get_local_id(1);
    const int i0 = get_group_id(1) * TS, j = get_group_id(0) * TS + lj;
    __local float As[TS][TS + 1];
    __local float Bs[TS][TS + 1];
    float acc[WPT];
    for (int w = 0; w < WPT; ++w) acc[w] = 0.0f;
    for (int t = 0; t < P; t += TS) {
        for (int w = 0; w < WPT; ++w) {
            const int r = li + w * RTS;
            const int i = i0 + r, pa = t + lj, pb = t + r;
            As[r][lj] = (i < M && pa < P) ? A[(long)i * sai + (long)pa * sap] : 0.0f;
            Bs[r][lj] = (pb < P && j < N) ? B[(long)pb * sbp + (long)j * sbj] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int q = 0; q < TS; ++q) {
            const float b = Bs[q][lj];
            for (int w = 0; w < WPT; ++w) acc[w] += As[li + w * RTS][q] * b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w = 0; w < WPT; ++w) {
        const int i = i0 + li + w * RTS;
        if (i < M && j < N) C[(long)i * N + j] += acc[w];
    }
}
)CL";

struct Gpu {
    std::mutex mu;  // one product at a time; busy means the caller uses the CPU
    Cl cl{};
    bool loaded = false, ready = false;
    GpuInfo info;
    cl_device_id dev = nullptr;
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    cl_kernel kern = nullptr;
    cl_mem buf[3] = {nullptr, nullptr, nullptr};
    size_t cap[3] = {0, 0, 0};
    double min_work = 0;  // products smaller than this (m * k * n) stay on the CPU
    std::string kernels_name;
    Kernels table{};
};

Gpu& G() {
    static Gpu g;
    return g;
}

template <class F>
bool sym(void* lib, const char* name, F& f) {
#if defined(_WIN32)
    f = reinterpret_cast<F>(reinterpret_cast<void*>(GetProcAddress((HMODULE)lib, name)));
#else
    f = reinterpret_cast<F>(dlsym(lib, name));
#endif
    return f != nullptr;
}

bool load_cl(Cl& cl, std::string& err) {
#if defined(_WIN32)
    // only the driver's loader in System32, never a DLL that sits next to the program
    void* lib = (void*)LoadLibraryExW(L"OpenCL.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#elif defined(__APPLE__)
    void* lib = dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL", RTLD_NOW);
#else
    void* lib = dlopen("libOpenCL.so.1", RTLD_NOW);
    if (!lib) lib = dlopen("libOpenCL.so", RTLD_NOW);
#endif
    if (!lib) {
        err = "OpenCL is not installed (it comes with the graphics driver)";
        return false;
    }
    bool ok = sym(lib, "clGetPlatformIDs", cl.GetPlatformIDs) && sym(lib, "clGetPlatformInfo", cl.GetPlatformInfo) &&
              sym(lib, "clGetDeviceIDs", cl.GetDeviceIDs) && sym(lib, "clGetDeviceInfo", cl.GetDeviceInfo) &&
              sym(lib, "clCreateContext", cl.CreateContext) && sym(lib, "clCreateCommandQueue", cl.CreateCommandQueue) &&
              sym(lib, "clCreateProgramWithSource", cl.CreateProgramWithSource) && sym(lib, "clBuildProgram", cl.BuildProgram) &&
              sym(lib, "clGetProgramBuildInfo", cl.GetProgramBuildInfo) && sym(lib, "clCreateKernel", cl.CreateKernel) &&
              sym(lib, "clCreateBuffer", cl.CreateBuffer) && sym(lib, "clReleaseMemObject", cl.ReleaseMemObject) &&
              sym(lib, "clEnqueueWriteBuffer", cl.EnqueueWriteBuffer) && sym(lib, "clEnqueueReadBuffer", cl.EnqueueReadBuffer) &&
              sym(lib, "clSetKernelArg", cl.SetKernelArg) && sym(lib, "clEnqueueNDRangeKernel", cl.EnqueueNDRangeKernel) &&
              sym(lib, "clFinish", cl.Finish);
    if (!ok) err = "the OpenCL library is incomplete";
    return ok;
}

std::string info_str(Gpu& g, cl_device_id d, cl_uint what) {
    char s[256] = {0};
    g.cl.GetDeviceInfo(d, what, sizeof s - 1, s, nullptr);
    return s;
}

// Picks the GPU with the most memory and compiles the kernel for it.
void init(Gpu& g) {
    if (g.loaded) return;
    g.loaded = true;
    GpuInfo& in = g.info;
    if (!load_cl(g.cl, in.error)) return;
    Cl& cl = g.cl;
    cl_uint np = 0;
    if (cl.GetPlatformIDs(0, nullptr, &np) != 0 || np == 0) {
        in.error = "no OpenCL platform";
        return;
    }
    std::vector<cl_platform_id> plats(np);
    cl.GetPlatformIDs(np, plats.data(), nullptr);
    cl_ulong best_mem = 0;
    for (cl_platform_id p : plats) {
        cl_uint nd = 0;
        if (cl.GetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != 0 || nd == 0) continue;
        std::vector<cl_device_id> devs(nd);
        cl.GetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr);
        for (cl_device_id d : devs) {
            cl_ulong mem = 0;
            size_t wg = 0;
            cl.GetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof mem, &mem, nullptr);
            cl.GetDeviceInfo(d, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof wg, &wg, nullptr);
            if (wg < 256 || mem <= best_mem) continue;
            best_mem = mem;
            g.dev = d;
            char pn[256] = {0};
            cl.GetPlatformInfo(p, CL_PLATFORM_NAME, sizeof pn - 1, pn, nullptr);
            in.platform = pn;
        }
    }
    if (!g.dev) {
        in.error = "no OpenCL graphics card found";
        return;
    }
    in.name = info_str(g, g.dev, CL_DEVICE_NAME);
    in.vendor = info_str(g, g.dev, CL_DEVICE_VENDOR);
    in.mem_gb = (double)best_mem / (1024.0 * 1024 * 1024);
    cl_int e = 0;
    g.ctx = cl.CreateContext(nullptr, 1, &g.dev, nullptr, nullptr, &e);
    if (e == 0) g.q = cl.CreateCommandQueue(g.ctx, g.dev, 0, &e);
    cl_program prog = nullptr;
    if (e == 0) prog = cl.CreateProgramWithSource(g.ctx, 1, &kSource, nullptr, &e);
    if (e == 0 && cl.BuildProgram(prog, 1, &g.dev, "", nullptr, nullptr) != 0) {
        char log[2048] = {0};
        cl.GetProgramBuildInfo(prog, g.dev, CL_PROGRAM_BUILD_LOG, sizeof log - 1, log, nullptr);
        in.error = std::string("the GPU program does not compile: ") + log;
        return;
    }
    if (e == 0) g.kern = cl.CreateKernel(prog, "gemm", &e);
    if (e != 0) {
        in.error = "OpenCL error " + std::to_string(e);
        return;
    }
    in.found = true;
    g.ready = true;
}

bool ensure(Gpu& g, int i, size_t bytes) {
    if (g.cap[i] >= bytes) return true;
    size_t want = std::max(bytes, g.cap[i] * 3 / 2);
    if (g.buf[i]) g.cl.ReleaseMemObject(g.buf[i]);
    g.buf[i] = nullptr;
    g.cap[i] = 0;
    cl_int e = 0;
    g.buf[i] = g.cl.CreateBuffer(g.ctx, CL_MEM_READ_WRITE, want, nullptr, &e);
    if (e != 0) {
        g.buf[i] = nullptr;
        return false;
    }
    g.cap[i] = want;
    return true;
}

// Runs one product; the caller holds g.mu.
cl_int run(Gpu& g, int v, const float* A, const float* B, float* C, int x, int y, int z) {
    int M, N, P, sai, sap, sbp, sbj;
    size_t na, nb, nc;
    if (v == 0) {  // mm(A x*y, B y*z, C x*z)
        M = x; P = y; N = z; sai = y; sap = 1; sbp = z; sbj = 1;
        na = (size_t)x * y; nb = (size_t)y * z; nc = (size_t)x * z;
    } else if (v == 1) {  // mm_ta(A x*y, B x*z, C y*z)
        M = y; P = x; N = z; sai = 1; sap = y; sbp = z; sbj = 1;
        na = (size_t)x * y; nb = (size_t)x * z; nc = (size_t)y * z;
    } else {  // mm_tb(A x*y, B z*y, C x*z)
        M = x; P = y; N = z; sai = y; sap = 1; sbp = 1; sbj = y;
        na = (size_t)x * y; nb = (size_t)z * y; nc = (size_t)x * z;
    }
    if (!ensure(g, 0, na * 4) || !ensure(g, 1, nb * 4) || !ensure(g, 2, nc * 4)) return -4;  // CL_MEM_OBJECT_ALLOCATION_FAILURE
    Cl& cl = g.cl;
    cl_int e = cl.EnqueueWriteBuffer(g.q, g.buf[0], 0, 0, na * 4, A, 0, nullptr, nullptr);
    if (!e) e = cl.EnqueueWriteBuffer(g.q, g.buf[1], 0, 0, nb * 4, B, 0, nullptr, nullptr);
    if (!e) e = cl.EnqueueWriteBuffer(g.q, g.buf[2], 0, 0, nc * 4, C, 0, nullptr, nullptr);
    int args[] = {M, N, P, sai, sap, sbp, sbj};
    if (!e) e = cl.SetKernelArg(g.kern, 0, sizeof(int), &args[0]);
    if (!e) e = cl.SetKernelArg(g.kern, 1, sizeof(int), &args[1]);
    if (!e) e = cl.SetKernelArg(g.kern, 2, sizeof(int), &args[2]);
    if (!e) e = cl.SetKernelArg(g.kern, 3, sizeof(cl_mem), &g.buf[0]);
    if (!e) e = cl.SetKernelArg(g.kern, 4, sizeof(int), &args[3]);
    if (!e) e = cl.SetKernelArg(g.kern, 5, sizeof(int), &args[4]);
    if (!e) e = cl.SetKernelArg(g.kern, 6, sizeof(cl_mem), &g.buf[1]);
    if (!e) e = cl.SetKernelArg(g.kern, 7, sizeof(int), &args[5]);
    if (!e) e = cl.SetKernelArg(g.kern, 8, sizeof(int), &args[6]);
    if (!e) e = cl.SetKernelArg(g.kern, 9, sizeof(cl_mem), &g.buf[2]);
    size_t global[2] = {(size_t)(N + 31) / 32 * 32, (size_t)(M + 31) / 32 * 8};
    size_t local[2] = {32, 8};
    if (!e) e = cl.EnqueueNDRangeKernel(g.q, g.kern, 2, nullptr, global, local, 0, nullptr, nullptr);
    if (!e) e = cl.EnqueueReadBuffer(g.q, g.buf[2], 1, 0, nc * 4, C, 0, nullptr, nullptr);
    if (e) cl.Finish(g.q);
    return e;
}

std::atomic<bool> g_on{false};

// Large products go to the GPU when it is free; anything else, or an error, uses the CPU.
bool offload(int v, const float* A, const float* B, float* C, int x, int y, int z) {
    Gpu& g = G();
    if (!g_on.load(std::memory_order_relaxed) || (double)x * y * z < g.min_work) return false;
    std::unique_lock<std::mutex> lk(g.mu, std::try_to_lock);
    if (!lk.owns_lock()) return false;
    if (run(g, v, A, B, C, x, y, z) == 0) return true;
    g_on = false;
    set_kernels(nullptr);
    return false;
}

void g_mm(const float* A, const float* B, float* C, int m, int k, int n) {
    if (!offload(0, A, B, C, m, k, n)) cpu_kernels().mm(A, B, C, m, k, n);
}
void g_mm_ta(const float* A, const float* B, float* C, int m, int k, int n) {
    if (!offload(1, A, B, C, m, k, n)) cpu_kernels().mm_ta(A, B, C, m, k, n);
}
void g_mm_tb(const float* A, const float* B, float* C, int m, int n, int k) {
    if (!offload(2, A, B, C, m, n, k)) cpu_kernels().mm_tb(A, B, C, m, n, k);
}

void cpu_gemm(int v, const float* A, const float* B, float* C, int x, int y, int z) {
    const Kernels& k = cpu_kernels();
    if (v == 0) k.mm(A, B, C, x, y, z);
    else if (v == 1) k.mm_ta(A, B, C, x, y, z);
    else k.mm_tb(A, B, C, x, y, z);
}

}  // namespace

const GpuInfo& gpu_probe() {
    Gpu& g = G();
    std::lock_guard<std::mutex> lk(g.mu);
    init(g);
    return g.info;
}

bool gpu_gemm(int variant, const float* A, const float* B, float* C, int x, int y, int z, std::string* err) {
    Gpu& g = G();
    std::lock_guard<std::mutex> lk(g.mu);
    init(g);
    if (!g.ready) {
        if (err) *err = g.info.error;
        return false;
    }
    cl_int e = run(g, variant, A, B, C, x, y, z);
    if (e && err) *err = "OpenCL error " + std::to_string(e);
    return e == 0;
}

bool gpu_active() { return g_on; }

bool gpu_use(bool on, std::string* msg) {
    Gpu& g = G();
    if (!on) {
        g_on = false;
        set_kernels(nullptr);
        return true;
    }
    const GpuInfo& in = gpu_probe();
    if (!in.found) {
        if (msg) *msg = in.error;
        return false;
    }
    std::string err;
    // Same results as the CPU (odd sizes cover the tile edges)?
    unsigned s = 12345;
    auto rnd = [&]() { s = s * 1103515245u + 12345u; return (float)((s >> 9) & 0xFFFF) / 32768.f - 1.f; };
    for (int v = 0; v < 3; ++v) {
        int x = 67, y = 45, z = 83;
        std::vector<float> A((size_t)x * std::max(y, z) * 2), B((size_t)std::max(x, z) * std::max(y, z) * 2), C1((size_t)x * z * 4), C2;
        for (float& f : A) f = rnd();
        for (float& f : B) f = rnd();
        for (float& f : C1) f = rnd();
        C2 = C1;
        cpu_gemm(v, A.data(), B.data(), C1.data(), x, y, z);
        if (!gpu_gemm(v, A.data(), B.data(), C2.data(), x, y, z, &err)) {
            if (msg) *msg = err;
            return false;
        }
        for (size_t i = 0; i < C1.size(); ++i)
            if (std::fabs(C1[i] - C2[i]) > 1e-3f * (1.f + std::fabs(C1[i]))) {
                if (msg) *msg = "the GPU results differ from the CPU";
                return false;
            }
    }
    // From which size is the GPU faster (copies included)?
    using clock = std::chrono::steady_clock;
    g.min_work = -1;
    double speedup = 0;
    for (int n : {128, 256, 384, 512, 768}) {
        std::vector<float> A((size_t)n * n), B((size_t)n * n), C((size_t)n * n, 0.f);
        for (float& f : A) f = rnd();
        for (float& f : B) f = rnd();
        auto best = [&](bool gpu) {
            double t = 1e9;
            for (int r = 0; r < 3; ++r) {
                auto t0 = clock::now();
                if (gpu) gpu_gemm(0, A.data(), B.data(), C.data(), n, n, n, nullptr);
                else cpu_gemm(0, A.data(), B.data(), C.data(), n, n, n);
                t = std::min(t, std::chrono::duration<double>(clock::now() - t0).count());
            }
            return t;
        };
        double tc = best(false), tg = best(true);
        if (tg < tc * 0.9 && g.min_work < 0) g.min_work = (double)n * n * n;
        if (tg < tc * 0.9 && n == 768) speedup = tc / tg;
        if (tg >= tc * 0.9 && n == 768) g.min_work = -1;  // only small sizes won: noise
    }
    if (g.min_work < 0) {
        if (msg) *msg = in.name + " is not faster than the CPU here, training stays on the CPU";
        return false;
    }
    g.kernels_name = "OpenCL " + in.name + " + " + cpu_kernels().name;
    g.table = cpu_kernels();
    g.table.name = g.kernels_name.c_str();
    g.table.mm = g_mm;
    g.table.mm_ta = g_mm_ta;
    g.table.mm_tb = g_mm_tb;
    g_on = true;
    set_kernels(&g.table);
    if (msg) {
        char b[256];
        std::snprintf(b, sizeof b, "%s: large matrix products on the GPU (%.1fx faster at 768)", in.name.c_str(), speedup);
        *msg = b;
    }
    return true;
}

}  // namespace kor
