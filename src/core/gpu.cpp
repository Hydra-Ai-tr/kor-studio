#include "gpu.h"

#include "kernels.h"
#include "parallel.h"
#include "tensor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
const cl_uint CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010;
const cl_uint CL_DEVICE_NAME = 0x102B;
const cl_uint CL_DEVICE_VENDOR = 0x102C;
const cl_uint CL_PROGRAM_BUILD_LOG = 0x1183;
const cl_ulong CL_MEM_READ_WRITE = 1;
const cl_ulong CL_MEM_ALLOC_HOST_PTR = 1 << 4;
const cl_ulong CL_MAP_READ = 1, CL_MAP_WRITE = 2;
const cl_int CL_MEM_OBJECT_ALLOCATION_FAILURE = -4, CL_OUT_OF_RESOURCES = -5, CL_INVALID_BUFFER_SIZE = -61;

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
    void*(CLAPI* EnqueueMapBuffer)(cl_command_queue, cl_mem, cl_uint, cl_ulong, size_t, size_t, cl_uint, const cl_event*, cl_event*,
                                   cl_int*);
    cl_int(CLAPI* EnqueueUnmapMemObject)(cl_command_queue, cl_mem, void*, cl_uint, const cl_event*, cl_event*);
    cl_int(CLAPI* SetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
    cl_int(CLAPI* EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint,
                                        const cl_event*, cl_event*);
    cl_int(CLAPI* Finish)(cl_command_queue);
};

// bgemm: for every z of the batch, C' = alpha * A' * B' (+ bias[j]) (+ C' when
// beta), where A'(i, p) = A[i * sai + p * sap], B'(p, j) = B[p * sbp + j * sbj]
// and C'(i, j) = C[i * sci + j * scj]; the matrices of item z start at
// off + (z / Hn) * bo + (z % Hn) * bi. One 64 x 64 tile per work-group of
// 16 x 16, every work-item sums a 4 x 4 block. `causal` skips the work of a
// lower-triangular (j <= i) problem: 1 = the output is only needed for j <= i,
// 2 = A' is zero for p > i, 3 = A' is zero for p < i.
//
// The row kernels use one work-group of 64 per row.
const char* kSource = R"CL(
#define TM 64
#define TK 16
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void bgemm(const int M, const int N, const int P, const int Hn,
           __global const float* A, const int aoff, const int sai, const int sap, const int abo, const int abi,
           __global const float* B, const int boff, const int sbp, const int sbj, const int bbo, const int bbi,
           __global float* C, const int coff, const int sci, const int scj, const int cbo, const int cbi,
           __global const float* bias, const int has_bias, const float alpha, const int beta, const int causal) {
    const int z = get_group_id(2), zo = z / Hn, zi = z % Hn;
    A += aoff + zo * abo + zi * abi;
    B += boff + zo * bbo + zi * bbi;
    C += coff + zo * cbo + zi * cbi;
    const int tx = get_local_id(0), ty = get_local_id(1), tid = ty * 16 + tx;
    const int i0 = get_group_id(1) * TM, j0 = get_group_id(0) * TM;
    if (causal == 1 && j0 > i0 + TM - 1) return;
    int p0 = 0, p1 = P;
    if (causal == 2) p1 = min(P, i0 + TM);
    if (causal == 3) p0 = i0;
    __local float As[TK][TM + 1];
    __local float Bs[TK][TM + 1];
    float acc[4][4];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) acc[r][c] = 0.0f;
    for (int t = p0; t < p1; t += TK) {
        for (int l = 0; l < 4; ++l) {
            const int e = tid + l * 256;
            int ii, pp, jj, qq;
            if (sap == 1) { ii = e >> 4; pp = e & 15; } else { ii = e & 63; pp = e >> 6; }
            if (sbj == 1) { jj = e & 63; qq = e >> 6; } else { jj = e >> 4; qq = e & 15; }
            const int gi = i0 + ii, gp = t + pp, gj = j0 + jj, gq = t + qq;
            As[pp][ii] = (gi < M && gp < p1) ? A[gi * sai + gp * sap] : 0.0f;
            Bs[qq][jj] = (gj < N && gq < p1) ? B[gq * sbp + gj * sbj] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int q = 0; q < TK; ++q) {
            float a[4], b[4];
            for (int r = 0; r < 4; ++r) a[r] = As[q][ty + 16 * r];
            for (int c = 0; c < 4; ++c) b[c] = Bs[q][tx + 16 * c];
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c) acc[r][c] = mad(a[r], b[c], acc[r][c]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int r = 0; r < 4; ++r) {
        const int i = i0 + ty + 16 * r;
        if (i >= M) break;
        for (int c = 0; c < 4; ++c) {
            const int j = j0 + tx + 16 * c;
            if (j >= N) break;
            __global float* d = C + i * sci + j * scj;
            float v = alpha * acc[r][c];
            if (has_bias) v += bias[j];
            *d = beta ? *d + v : v;
        }
    }
}

#define WG 64
float wg_sum(__local float* red, float v) {
    const int l = get_local_id(0);
    red[l] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (l < s) red[l] += red[l + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    v = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return v;
}
float wg_max(__local float* red, float v) {
    const int l = get_local_id(0);
    red[l] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (l < s) red[l] = fmax(red[l], red[l + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    v = red[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return v;
}

// Row-wise softmax over j <= i of the scores (row r is position i = r % T); zero above.
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void softmax_causal(__global float* S, const int T) {
    __local float red[WG];
    const long row = get_group_id(0);
    const int i = (int)(row % T), l = get_local_id(0);
    __global float* s = S + row * T;
    float mx = -INFINITY;
    for (int j = l; j <= i; j += WG) mx = fmax(mx, s[j]);
    mx = wg_max(red, mx);
    float sum = 0.0f;
    for (int j = l; j <= i; j += WG) {
        const float e = exp(s[j] - mx);
        s[j] = e;
        sum += e;
    }
    const float is = 1.0f / wg_sum(red, sum);
    for (int j = l; j < T; j += WG) s[j] = j <= i ? s[j] * is : 0.0f;
}

// D = scale * P * (D - sum_j P * D) over j <= i, zero above (softmax backward in place).
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void softmax_back(__global const float* Pm, __global float* D, const int T, const float scale) {
    __local float red[WG];
    const long row = get_group_id(0);
    const int i = (int)(row % T), l = get_local_id(0);
    __global const float* p = Pm + row * T;
    __global float* d = D + row * T;
    float dot = 0.0f;
    for (int j = l; j <= i; j += WG) dot += p[j] * d[j];
    dot = wg_sum(red, dot);
    for (int j = l; j < T; j += WG) d[j] = j <= i ? scale * p[j] * (d[j] - dot) : 0.0f;
}

__kernel void fill(__global float* x, const int n, const float v) {
    const int i = get_global_id(0);
    if (i < n) x[i] = v;
}

__kernel void add_to(__global const float* a, __global float* b, const int n) {
    const int i = get_global_id(0);
    if (i < n) b[i] += a[i];
}

__kernel void embed(__global const int* ids, __global const float* tok, __global const float* pos, __global float* x,
                    const int T, const int C, const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    const int r = i / C, j = i % C;
    x[i] = tok[ids[r] * C + j] + pos[(r % T) * C + j];
}

// y = g * (x - mean) / sqrt(var + 1e-5) + b; st keeps (mean, 1 / std) per row.
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void layernorm(__global const float* x, __global const float* g, __global const float* b, __global float* y,
               __global float* st, const int C) {
    __local float red[WG];
    const int row = get_group_id(0), l = get_local_id(0);
    __global const float* a = x + row * C;
    __global float* o = y + row * C;
    float s = 0.0f;
    for (int j = l; j < C; j += WG) s += a[j];
    const float mean = wg_sum(red, s) / C;
    s = 0.0f;
    for (int j = l; j < C; j += WG) s += (a[j] - mean) * (a[j] - mean);
    const float is = 1.0f / sqrt(wg_sum(red, s) / C + 1e-5f);
    for (int j = l; j < C; j += WG) o[j] = g[j] * (a[j] - mean) * is + b[j];
    if (l == 0) { st[2 * row] = mean; st[2 * row + 1] = is; }
}

// dx of the layer norm above; dyxh = dy * xhat (summed over rows later for dg).
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void layernorm_back(__global const float* dy, __global const float* x, __global const float* st, __global const float* g,
                    __global float* dx, __global float* dyxh, const int C) {
    __local float red[WG];
    const int row = get_group_id(0), l = get_local_id(0);
    const float mean = st[2 * row], is = st[2 * row + 1];
    __global const float* d = dy + row * C;
    __global const float* a = x + row * C;
    float m1 = 0.0f, m2 = 0.0f;
    for (int j = l; j < C; j += WG) {
        const float xh = (a[j] - mean) * is, dxh = d[j] * g[j];
        m1 += dxh;
        m2 += dxh * xh;
    }
    m1 = wg_sum(red, m1) / C;
    m2 = wg_sum(red, m2) / C;
    for (int j = l; j < C; j += WG) {
        const float xh = (a[j] - mean) * is;
        dx[row * C + j] = is * (d[j] * g[j] - m1 - xh * m2);
        dyxh[row * C + j] = d[j] * xh;
    }
}

#define GELU_C 0.7978845608f
__kernel void gelu(__global const float* a, __global float* o, const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    const float v = a[i];
    o[i] = 0.5f * v * (1.0f + tanh(GELU_C * (v + 0.044715f * v * v * v)));
}

__kernel void gelu_back(__global const float* a, __global float* d, const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    const float v = a[i];
    const float t = tanh(GELU_C * (v + 0.044715f * v * v * v));
    const float dt = (1.0f - t * t) * GELU_C * (1.0f + 3.0f * 0.044715f * v * v);
    d[i] *= 0.5f * (1.0f + t) + 0.5f * v * dt;
}

// Dropout keeps element i with probability 1 - p (the same mask for the same seed).
float keep_scale(uint seed, uint i, float p) {
    if (p <= 0.0f) return 1.0f;
    uint h = i * 0x9E3779B1u ^ seed;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return (float)(h >> 8) * (1.0f / 16777216.0f) >= p ? 1.0f / (1.0f - p) : 0.0f;
}

// y = x + dropout(a)
__kernel void resid_drop(__global const float* x, __global const float* a, __global float* y, const int n, const float p,
                         const uint seed) {
    const int i = get_global_id(0);
    if (i < n) y[i] = x[i] + a[i] * keep_scale(seed, (uint)i, p);
}

// o = dropout'(d) with the mask of resid_drop
__kernel void drop_back(__global const float* d, __global float* o, const int n, const float p, const uint seed) {
    const int i = get_global_id(0);
    if (i < n) o[i] = d[i] * keep_scale(seed, (uint)i, p);
}

// Cross entropy per row: out = (loss, 1 if the arg max is the target); with
// grad the logits become dL/dz * gs (gs = base * weight of the row).
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void xent(__global float* z, __global const int* tgt, __global const float* w, __global float* out, const int V,
          const float base, const int use_w, const int grad) {
    __local float red[WG];
    __local int redi[WG];
    const int row = get_group_id(0), l = get_local_id(0);
    __global float* zr = z + (long)row * V;
    float mx = -INFINITY;
    int arg = V;
    for (int j = l; j < V; j += WG)
        if (zr[j] > mx) { mx = zr[j]; arg = j; }
    red[l] = mx;
    redi[l] = arg;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (l < s && (red[l + s] > red[l] || (red[l + s] == red[l] && redi[l + s] < redi[l]))) {
            red[l] = red[l + s];
            redi[l] = redi[l + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    mx = red[0];
    arg = redi[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float s = 0.0f;
    for (int j = l; j < V; j += WG) s += exp(zr[j] - mx);
    const float sum = wg_sum(red, s);
    const int t = tgt[row];
    if (l == 0) {
        out[2 * row] = log(sum) - (zr[t] - mx);
        out[2 * row + 1] = arg == t ? 1.0f : 0.0f;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (!grad) return;
    const float gs = base * (use_w ? w[row] : 1.0f), is = 1.0f / sum;
    for (int j = l; j < V; j += WG) zr[j] = exp(zr[j] - mx) * is * gs - (j == t ? gs : 0.0f);
}
)CL";

// A device buffer that grows on demand.
struct DBuf {
    cl_mem m = nullptr;
    size_t cap = 0;  // bytes
};

// Device memory of the whole-model text path (one set per shape).
struct GptBufs {
    std::vector<DBuf> w, g;  // parameters and their gradients
    // per layer: block input, ln1 out + stats, qkv, attention out, residual h, ln2 out + stats, fc1 out
    std::vector<DBuf> x, a1, st1, qkv, ho, h, a3, st2, a4;
    DBuf xf, stf, logits, ids, tgt, wt, rows, ones;
    DBuf s1, s4, S, dS, dy, dx, t1, t2, t3, yx, dq, d4;
};

struct Gpu {
    std::mutex mu;  // one job at a time; busy means the caller uses the CPU
    Cl cl{};
    bool loaded = false, ready = false;
    GpuInfo info;
    cl_device_id dev = nullptr;
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    cl_kernel kgemm = nullptr, ksoft = nullptr, ksoftb = nullptr, kfill = nullptr, kadd = nullptr, kembed = nullptr, kln = nullptr,
              klnb = nullptr, kgelu = nullptr, kgelub = nullptr, kresid = nullptr, kdropb = nullptr, kxent = nullptr;
    // products and attention of the CPU layers
    DBuf pa, pb, pc, aq, as, ad, ao, adq;
    GptBufs gpt;
    // page-locked host memory: copies to and from it run at full bus speed
    cl_mem pin = nullptr;
    float* pinned = nullptr;
    size_t pin_cap = 0;
    size_t max_alloc = 0, mem = 0;
    double min_work = 0;        // products smaller than this (m * k * n) stay on the CPU
    double attn_min = 1 << 21;  // attention below this (B * H * T * T * hd) stays on the CPU
    bool attn_ok = false;
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
              sym(lib, "clEnqueueMapBuffer", cl.EnqueueMapBuffer) && sym(lib, "clEnqueueUnmapMemObject", cl.EnqueueUnmapMemObject) &&
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

// Picks the GPU with the most memory and compiles the kernels for it.
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
    g.mem = (size_t)best_mem;
    cl_ulong ma = 0;
    cl.GetDeviceInfo(g.dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof ma, &ma, nullptr);
    g.max_alloc = ma ? (size_t)ma : (size_t)(best_mem / 4);
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
    struct { cl_kernel* k; const char* name; } ks[] = {
        {&g.kgemm, "bgemm"}, {&g.ksoft, "softmax_causal"}, {&g.ksoftb, "softmax_back"}, {&g.kfill, "fill"},
        {&g.kadd, "add_to"}, {&g.kembed, "embed"}, {&g.kln, "layernorm"}, {&g.klnb, "layernorm_back"},
        {&g.kgelu, "gelu"}, {&g.kgelub, "gelu_back"}, {&g.kresid, "resid_drop"}, {&g.kdropb, "drop_back"},
        {&g.kxent, "xent"}};
    for (auto& k : ks)
        if (e == 0) *k.k = cl.CreateKernel(prog, k.name, &e);
    if (e != 0) {
        in.error = "OpenCL error " + std::to_string(e);
        return;
    }
    in.found = true;
    g.ready = true;
}

bool need(Gpu& g, DBuf& b, size_t floats) {
    size_t bytes = std::max<size_t>(floats, 1) * 4;
    if (b.cap >= bytes) return true;
    if (bytes > g.max_alloc) return false;
    size_t want = std::min(g.max_alloc, std::max(bytes, b.cap * 5 / 4));
    if (b.m) g.cl.ReleaseMemObject(b.m);
    b.m = nullptr;
    b.cap = 0;
    cl_int e = 0;
    b.m = g.cl.CreateBuffer(g.ctx, CL_MEM_READ_WRITE, want, nullptr, &e);
    if (e != 0) {
        b.m = nullptr;
        return false;
    }
    b.cap = want;
    return true;
}

void release(Gpu& g, DBuf& b) {
    if (b.m) g.cl.ReleaseMemObject(b.m);
    b = DBuf();
}

// A page-locked host area of at least `floats` floats (nullptr when the driver refuses).
float* pinned(Gpu& g, size_t floats) {
    size_t bytes = floats * 4;
    if (g.pin_cap >= bytes) return g.pinned;
    size_t want = std::max(bytes, g.pin_cap * 3 / 2);
    if (g.pin) {
        g.cl.EnqueueUnmapMemObject(g.q, g.pin, g.pinned, 0, nullptr, nullptr);
        g.cl.Finish(g.q);
        g.cl.ReleaseMemObject(g.pin);
    }
    g.pin = nullptr;
    g.pinned = nullptr;
    g.pin_cap = 0;
    if (want > g.max_alloc) return nullptr;
    cl_int e = 0;
    g.pin = g.cl.CreateBuffer(g.ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, want, nullptr, &e);
    if (e) {
        g.pin = nullptr;
        return nullptr;
    }
    void* p = g.cl.EnqueueMapBuffer(g.q, g.pin, 1, CL_MAP_READ | CL_MAP_WRITE, 0, want, 0, nullptr, nullptr, &e);
    if (e || !p) {
        g.cl.ReleaseMemObject(g.pin);
        g.pin = nullptr;
        return nullptr;
    }
    g.pinned = (float*)p;
    g.pin_cap = want;
    return g.pinned;
}

// Splits [0, n) into chunks for the CPU threads.
template <class F>
void chunks(size_t n, F f) {
    const size_t chunk = 1 << 16;
    parallel_for((int)((n + chunk - 1) / chunk), [&](int s, int e) {
        for (int c = s; c < e; ++c) f((size_t)c * chunk, std::min(n, (size_t)(c + 1) * chunk));
    });
}

// Blocking copy of n floats to the card (through the pinned area when there is one).
cl_int upload(Gpu& g, DBuf& b, const void* src, size_t n) {
    float* p = pinned(g, n);
    if (!p) return g.cl.EnqueueWriteBuffer(g.q, b.m, 1, 0, n * 4, src, 0, nullptr, nullptr);
    chunks(n, [&](size_t a, size_t z) { std::memcpy(p + a, (const float*)src + a, (z - a) * 4); });
    return g.cl.EnqueueWriteBuffer(g.q, b.m, 1, 0, n * 4, p, 0, nullptr, nullptr);
}

// Reads n floats; `add` sums them into dst instead of overwriting.
cl_int download(Gpu& g, DBuf& b, float* dst, size_t n, bool add) {
    float* p = pinned(g, n);
    if (!p && !add) return g.cl.EnqueueReadBuffer(g.q, b.m, 1, 0, n * 4, dst, 0, nullptr, nullptr);
    std::vector<float> tmp;
    if (!p) {
        tmp.resize(n);
        p = tmp.data();
    }
    cl_int e = g.cl.EnqueueReadBuffer(g.q, b.m, 1, 0, n * 4, p, 0, nullptr, nullptr);
    if (e) return e;
    chunks(n, [&](size_t a, size_t z) {
        if (add)
            for (size_t i = a; i < z; ++i) dst[i] += p[i];
        else
            std::memcpy(dst + a, p + a, (z - a) * 4);
    });
    return 0;
}

// Sets kernel arguments in order.
struct Args {
    Gpu& g;
    cl_kernel k;
    cl_int e = 0;
    cl_uint n = 0;
    Args(Gpu& g_, cl_kernel k_) : g(g_), k(k_) {}
    Args& i(int v) { if (!e) e = g.cl.SetKernelArg(k, n, sizeof v, &v); ++n; return *this; }
    Args& u(uint32_t v) { if (!e) e = g.cl.SetKernelArg(k, n, sizeof v, &v); ++n; return *this; }
    Args& f(float v) { if (!e) e = g.cl.SetKernelArg(k, n, sizeof v, &v); ++n; return *this; }
    Args& m(cl_mem v) { if (!e) e = g.cl.SetKernelArg(k, n, sizeof v, &v); ++n; return *this; }
    Args& m(DBuf& b) { return m(b.m); }
    // one work-item per element
    cl_int flat(size_t count) {
        size_t local = 256, global = (count + 255) / 256 * 256;
        if (!e) e = g.cl.EnqueueNDRangeKernel(g.q, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        return e;
    }
    // one work-group of 64 per row
    cl_int rows(size_t count) {
        size_t local = 64, global = count * 64;
        if (!e) e = g.cl.EnqueueNDRangeKernel(g.q, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        return e;
    }
};

// A strided matrix view inside a device buffer (see bgemm).
struct View {
    cl_mem m;
    int off, si, sj, bo, bi;
};
View rowmajor(DBuf& b, int cols, int off = 0) { return View{b.m, off, cols, 1, 0, 0}; }
View transposed(DBuf& b, int cols, int off = 0) { return View{b.m, off, 1, cols, 0, 0}; }

cl_int gemm(Gpu& g, int M, int N, int P, int batch, int Hn, View a, View b, View c, float alpha, int beta, int causal,
            cl_mem bias = nullptr) {
    Args s(g, g.kgemm);
    s.i(M).i(N).i(P).i(Hn);
    s.m(a.m).i(a.off).i(a.si).i(a.sj).i(a.bo).i(a.bi);
    s.m(b.m).i(b.off).i(b.si).i(b.sj).i(b.bo).i(b.bi);
    s.m(c.m).i(c.off).i(c.si).i(c.sj).i(c.bo).i(c.bi);
    s.m(bias).i(bias ? 1 : 0).f(alpha).i(beta).i(causal);
    size_t global[3] = {(size_t)(N + 63) / 64 * 16, (size_t)(M + 63) / 64 * 16, (size_t)batch};
    size_t local[3] = {16, 16, 1};
    if (!s.e) s.e = g.cl.EnqueueNDRangeKernel(g.q, s.k, 3, nullptr, global, local, 0, nullptr, nullptr);
    return s.e;
}

bool is_memory_error(cl_int e) {
    return e == CL_MEM_OBJECT_ALLOCATION_FAILURE || e == CL_OUT_OF_RESOURCES || e == CL_INVALID_BUFFER_SIZE;
}

// One product with the argument order of Kernels (C is added to); the caller holds g.mu.
cl_int run(Gpu& g, int v, const float* A, const float* B, float* C, int x, int y, int z) {
    int M, N, P;
    View a{nullptr, 0, 0, 0, 0, 0}, b{nullptr, 0, 0, 0, 0, 0};
    size_t na, nb, nc;
    if (v == 0) {  // mm(A x*y, B y*z, C x*z)
        M = x; P = y; N = z; a.si = y; a.sj = 1; b.si = z; b.sj = 1;
        na = (size_t)x * y; nb = (size_t)y * z; nc = (size_t)x * z;
    } else if (v == 1) {  // mm_ta(A x*y, B x*z, C y*z)
        M = y; P = x; N = z; a.si = 1; a.sj = y; b.si = z; b.sj = 1;
        na = (size_t)x * y; nb = (size_t)x * z; nc = (size_t)y * z;
    } else {  // mm_tb(A x*y, B z*y, C x*z)
        M = x; P = y; N = z; a.si = y; a.sj = 1; b.si = 1; b.sj = y;
        na = (size_t)x * y; nb = (size_t)z * y; nc = (size_t)x * z;
    }
    if (!need(g, g.pa, na) || !need(g, g.pb, nb) || !need(g, g.pc, nc)) return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    a.m = g.pa.m;
    b.m = g.pb.m;
    cl_int e = upload(g, g.pa, A, na);
    if (!e) e = upload(g, g.pb, B, nb);
    if (!e) e = gemm(g, M, N, P, 1, 1, a, b, rowmajor(g.pc, N), 1.f, 0, 0);
    if (!e) e = download(g, g.pc, C, nc, true);
    if (e) g.cl.Finish(g.q);
    return e;
}

// Causal self-attention of qkv [B*T x 3C] (C = H * hd) into out [B*T x C],
// everything already on the card; S receives the attention weights. Without
// `out` only the weights are computed.
cl_int attn_dev(Gpu& g, DBuf& qkv, DBuf& S, DBuf* out, int B, int T, int H, int hd) {
    const int C = H * hd, C3 = 3 * C;
    float scale = 1.f / std::sqrt((float)hd);
    // S = scale * Q K^T
    cl_int e = gemm(g, T, T, hd, B * H, H, View{qkv.m, 0, C3, 1, T * C3, hd}, View{qkv.m, C, 1, C3, T * C3, hd},
                    View{S.m, 0, T, 1, H * T * T, T * T}, scale, 0, 1);
    if (!e) e = Args(g, g.ksoft).m(S).i(T).rows((size_t)B * H * T);
    // O = P V
    if (!e && out)
        e = gemm(g, T, hd, T, B * H, H, View{S.m, 0, T, 1, H * T * T, T * T}, View{qkv.m, 2 * C, C3, 1, T * C3, hd},
                 View{out->m, 0, C, 1, T * C, hd}, 1.f, 0, 2);
    return e;
}

// Gradient of attn_dev: dout [B*T x C] -> dqkv [B*T x 3C] (overwritten). The
// weights are recomputed into S, so nothing stays on the card between the passes.
cl_int attn_dev_back(Gpu& g, DBuf& qkv, DBuf& S, DBuf& D, DBuf& dout, DBuf& dqkv, int B, int T, int H, int hd) {
    const int C = H * hd, C3 = 3 * C;
    float scale = 1.f / std::sqrt((float)hd);
    cl_int e = attn_dev(g, qkv, S, nullptr, B, T, H, hd);
    View Pt{S.m, 0, 1, T, H * T * T, T * T};
    View Dv{D.m, 0, T, 1, H * T * T, T * T}, Dt{D.m, 0, 1, T, H * T * T, T * T};
    View dO{dout.m, 0, C, 1, T * C, hd};
    // dP = dO V^T
    if (!e) e = gemm(g, T, T, hd, B * H, H, dO, View{qkv.m, 2 * C, 1, C3, T * C3, hd}, Dv, 1.f, 0, 1);
    // dV = P^T dO
    if (!e) e = gemm(g, T, hd, T, B * H, H, Pt, dO, View{dqkv.m, 2 * C, C3, 1, T * C3, hd}, 1.f, 0, 3);
    // dS = scale * P * (dP - rowsum(P * dP))
    if (!e) e = Args(g, g.ksoftb).m(S).m(D).i(T).f(scale).rows((size_t)B * H * T);
    // dQ = dS K, dK = dS^T Q
    if (!e) e = gemm(g, T, hd, T, B * H, H, Dv, View{qkv.m, C, C3, 1, T * C3, hd}, View{dqkv.m, 0, C3, 1, T * C3, hd}, 1.f, 0, 2);
    if (!e) e = gemm(g, T, hd, T, B * H, H, Dt, View{qkv.m, 0, C3, 1, T * C3, hd}, View{dqkv.m, C, C3, 1, T * C3, hd}, 1.f, 0, 3);
    return e;
}

// Attention of the CPU layers: copies in, runs, copies out. The caller holds g.mu.
cl_int attn_host(Gpu& g, bool backward, const float* qkv, const float* dout, float* res, int B, int T, int H, int hd) {
    const int C = H * hd;
    size_t nq = (size_t)B * T * 3 * C, ns = (size_t)B * H * T * T, no = (size_t)B * T * C;
    if (!need(g, g.aq, nq) || !need(g, g.as, ns) || !need(g, g.ao, no)) return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    if (backward && (!need(g, g.ad, ns) || !need(g, g.adq, nq))) return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    cl_int e = upload(g, g.aq, qkv, nq);
    if (!backward) {
        if (!e) e = attn_dev(g, g.aq, g.as, &g.ao, B, T, H, hd);
        if (!e) e = download(g, g.ao, res, no, false);
    } else {
        if (!e) e = upload(g, g.ao, dout, no);
        if (!e) e = attn_dev_back(g, g.aq, g.as, g.ad, g.ao, g.adq, B, T, H, hd);
        if (!e) e = download(g, g.adq, res, nq, false);
    }
    return e;
}

// ------------------------------------------------- whole text model on the card

uint32_t mix32(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
    return (uint32_t)x;
}

struct GptJob {
    const GpuGpt* m;
    const int* ids;
    const int* tgt;
    const float* w;
    int B, T;
    bool train;
    float dropout;
    uint64_t seed;
    float scale;
};

// Device memory (floats) the job needs, to refuse what would not fit.
double gpt_floats(const GpuGpt& m, int B, int T) {
    double N = (double)B * T, C = m.dim, L = m.nlayer;
    double params = 0;
    for (Param* p : m.params) params += (double)p->w.size();
    double per_layer = N * (C * 12 + 4);
    double att = 2.0 * B * m.heads * T * T;
    double scratch = N * (C * 6 + 4 * C * 2 + 3 * C) + N * m.vocab + N * 4;
    return params * 2 + per_layer * L + N * C * 2 + att + scratch;
}

cl_int gpt_run(Gpu& g, const GptJob& j, float* loss, int* correct) {
    const GpuGpt& m = *j.m;
    const int B = j.B, T = j.T, N = B * T, C = m.dim, H = m.heads, hd = C / H, L = m.nlayer, V = m.vocab, C3 = 3 * C, C4 = 4 * C;
    GptBufs& d = g.gpt;
    const std::vector<Param*>& ps = m.params;
    const size_t np = ps.size();
    auto fail = [&]() { return CL_MEM_OBJECT_ALLOCATION_FAILURE; };
    if (d.w.size() < np) { d.w.resize(np); d.g.resize(np); }
    for (auto* v : {&d.x, &d.a1, &d.st1, &d.qkv, &d.ho, &d.h, &d.a3, &d.st2, &d.a4})
        if ((int)v->size() < L + 1) v->resize(L + 1);
    for (size_t i = 0; i < np; ++i)
        if (!need(g, d.w[i], ps[i]->w.size()) || (j.train && !need(g, d.g[i], ps[i]->w.size()))) return fail();
    for (int l = 0; l <= L; ++l)
        if (!need(g, d.x[l], (size_t)N * C)) return fail();
    for (int l = 0; l < L; ++l)
        if (!need(g, d.a1[l], (size_t)N * C) || !need(g, d.st1[l], (size_t)N * 2) || !need(g, d.qkv[l], (size_t)N * C3) ||
            !need(g, d.ho[l], (size_t)N * C) || !need(g, d.h[l], (size_t)N * C) || !need(g, d.a3[l], (size_t)N * C) ||
            !need(g, d.st2[l], (size_t)N * 2) || !need(g, d.a4[l], (size_t)N * C4))
            return fail();
    if (!need(g, d.xf, (size_t)N * C) || !need(g, d.stf, (size_t)N * 2) || !need(g, d.logits, (size_t)N * V) ||
        !need(g, d.ids, N) || !need(g, d.tgt, N) || !need(g, d.wt, N) || !need(g, d.rows, (size_t)N * 2) || !need(g, d.ones, N) ||
        !need(g, d.s1, (size_t)N * C) || !need(g, d.s4, (size_t)N * C4) || !need(g, d.S, (size_t)B * H * T * T))
        return fail();
    if (j.train && (!need(g, d.dS, (size_t)B * H * T * T) || !need(g, d.dy, (size_t)N * C) || !need(g, d.dx, (size_t)N * C) ||
                    !need(g, d.t1, (size_t)N * C) || !need(g, d.t2, (size_t)N * C) || !need(g, d.t3, (size_t)N * C) ||
                    !need(g, d.yx, (size_t)N * C) || !need(g, d.dq, (size_t)N * C3) || !need(g, d.d4, (size_t)N * C4)))
        return fail();

    Cl& cl = g.cl;
    cl_int e = 0;
    // parameters in, gradients zeroed (copies run in order before the kernels)
    for (size_t i = 0; i < np && !e; ++i) {
        e = cl.EnqueueWriteBuffer(g.q, d.w[i].m, 0, 0, ps[i]->w.size() * 4, ps[i]->w.data(), 0, nullptr, nullptr);
        if (!e && j.train) e = Args(g, g.kfill).m(d.g[i]).i((int)ps[i]->w.size()).f(0.f).flat(ps[i]->w.size());
    }
    if (!e) e = cl.EnqueueWriteBuffer(g.q, d.ids.m, 0, 0, (size_t)N * 4, j.ids, 0, nullptr, nullptr);
    if (!e) e = cl.EnqueueWriteBuffer(g.q, d.tgt.m, 0, 0, (size_t)N * 4, j.tgt, 0, nullptr, nullptr);
    if (!e && j.w) e = cl.EnqueueWriteBuffer(g.q, d.wt.m, 0, 0, (size_t)N * 4, j.w, 0, nullptr, nullptr);
    if (!e) e = Args(g, g.kfill).m(d.ones).i(N).f(1.f).flat(N);

    // parameter index: tok, pos, then 12 per block, then lnf g, b, head W, b
    auto P = [&](int l, int k) -> DBuf& { return d.w[2 + l * 12 + k]; };
    auto Gd = [&](int l, int k) -> DBuf& { return d.g[2 + l * 12 + k]; };
    enum { LN1G, LN1B, QKVW, QKVB, PROJW, PROJB, LN2G, LN2B, FC1W, FC1B, FC2W, FC2B };
    const int tail = 2 + 12 * L;
    auto linear = [&](DBuf& x, int in, DBuf& W, DBuf& b, DBuf& y, int out) {
        return gemm(g, N, out, in, 1, 1, rowmajor(x, in), rowmajor(W, out), rowmajor(y, out), 1.f, 0, 0, b.m);
    };
    // dW += x^T dy, db += sum of dy over rows, dx = dy W^T (when dx)
    auto linear_back = [&](DBuf& x, int in, DBuf& W, DBuf& dW, DBuf& db, DBuf& dy, int out, DBuf* dx) {
        cl_int r = gemm(g, in, out, N, 1, 1, transposed(x, in), rowmajor(dy, out), rowmajor(dW, out), 1.f, 1, 0);
        if (!r) r = gemm(g, 1, out, N, 1, 1, View{d.ones.m, 0, 0, 1, 0, 0}, rowmajor(dy, out), rowmajor(db, out), 1.f, 1, 0);
        if (!r && dx) r = gemm(g, N, in, out, 1, 1, rowmajor(dy, out), transposed(W, out), rowmajor(*dx, in), 1.f, 0, 0);
        return r;
    };
    auto layernorm = [&](DBuf& x, DBuf& gw, DBuf& bw, DBuf& y, DBuf& st) {
        return Args(g, g.kln).m(x).m(gw).m(bw).m(y).m(st).i(C).rows(N);
    };
    // dx = LN'(dy); dg, db accumulate
    auto layernorm_back = [&](DBuf& dy, DBuf& x, DBuf& st, DBuf& gw, DBuf& dg, DBuf& db, DBuf& dx) {
        cl_int r = Args(g, g.klnb).m(dy).m(x).m(st).m(gw).m(dx).m(d.yx).i(C).rows(N);
        View ones{d.ones.m, 0, 0, 1, 0, 0};
        if (!r) r = gemm(g, 1, C, N, 1, 1, ones, rowmajor(d.yx, C), rowmajor(dg, C), 1.f, 1, 0);
        if (!r) r = gemm(g, 1, C, N, 1, 1, ones, rowmajor(dy, C), rowmajor(db, C), 1.f, 1, 0);
        return r;
    };
    const float p = j.train ? j.dropout : 0.f;
    auto seed_of = [&](int l, int site) { return mix32(j.seed * 0x9E3779B97F4A7C15ull + (uint64_t)l * 2 + site); };
    const int NC = N * C;

    // forward
    if (!e) e = Args(g, g.kembed).m(d.ids).m(d.w[0]).m(d.w[1]).m(d.x[0]).i(T).i(C).i(NC).flat(NC);
    for (int l = 0; l < L && !e; ++l) {
        e = layernorm(d.x[l], P(l, LN1G), P(l, LN1B), d.a1[l], d.st1[l]);
        if (!e) e = linear(d.a1[l], C, P(l, QKVW), P(l, QKVB), d.qkv[l], C3);
        if (!e) e = attn_dev(g, d.qkv[l], d.S, &d.ho[l], B, T, H, hd);
        if (!e) e = linear(d.ho[l], C, P(l, PROJW), P(l, PROJB), d.s1, C);
        if (!e) e = Args(g, g.kresid).m(d.x[l]).m(d.s1).m(d.h[l]).i(NC).f(p).u(seed_of(l, 0)).flat(NC);
        if (!e) e = layernorm(d.h[l], P(l, LN2G), P(l, LN2B), d.a3[l], d.st2[l]);
        if (!e) e = linear(d.a3[l], C, P(l, FC1W), P(l, FC1B), d.a4[l], C4);
        if (!e) e = Args(g, g.kgelu).m(d.a4[l]).m(d.s4).i(N * C4).flat((size_t)N * C4);
        if (!e) e = linear(d.s4, C4, P(l, FC2W), P(l, FC2B), d.s1, C);
        if (!e) e = Args(g, g.kresid).m(d.h[l]).m(d.s1).m(d.x[l + 1]).i(NC).f(p).u(seed_of(l, 1)).flat(NC);
    }
    if (!e) e = layernorm(d.x[L], d.w[tail], d.w[tail + 1], d.xf, d.stf);
    if (!e) e = linear(d.xf, C, d.w[tail + 2], d.w[tail + 3], d.logits, V);
    double wsum = N;
    if (j.w) {
        wsum = 0;
        for (int r = 0; r < N; ++r) wsum += j.w[r];
        if (wsum <= 0) wsum = 1;
    }
    if (!e) e = Args(g, g.kxent).m(d.logits).m(d.tgt).m(d.wt).m(d.rows).i(V).f((float)(j.scale / wsum)).i(j.w ? 1 : 0).i(j.train ? 1 : 0).rows(N);

    // backward
    if (j.train && !e) {
        e = linear_back(d.xf, C, d.w[tail + 2], d.g[tail + 2], d.g[tail + 3], d.logits, V, &d.t1);
        if (!e) e = layernorm_back(d.t1, d.x[L], d.stf, d.w[tail], d.g[tail], d.g[tail + 1], d.dy);
        for (int l = L - 1; l >= 0 && !e; --l) {
            // dy: gradient of the block output
            e = Args(g, g.kdropb).m(d.dy).m(d.t1).i(NC).f(p).u(seed_of(l, 1)).flat(NC);
            if (!e) e = Args(g, g.kgelu).m(d.a4[l]).m(d.s4).i(N * C4).flat((size_t)N * C4);
            if (!e) e = linear_back(d.s4, C4, P(l, FC2W), Gd(l, FC2W), Gd(l, FC2B), d.t1, C, &d.d4);
            if (!e) e = Args(g, g.kgelub).m(d.a4[l]).m(d.d4).i(N * C4).flat((size_t)N * C4);
            if (!e) e = linear_back(d.a3[l], C, P(l, FC1W), Gd(l, FC1W), Gd(l, FC1B), d.d4, C4, &d.t2);
            if (!e) e = layernorm_back(d.t2, d.h[l], d.st2[l], P(l, LN2G), Gd(l, LN2G), Gd(l, LN2B), d.t3);
            if (!e) e = Args(g, g.kadd).m(d.dy).m(d.t3).i(NC).flat(NC);  // t3 = dh
            if (!e) e = Args(g, g.kdropb).m(d.t3).m(d.t1).i(NC).f(p).u(seed_of(l, 0)).flat(NC);
            if (!e) e = linear_back(d.ho[l], C, P(l, PROJW), Gd(l, PROJW), Gd(l, PROJB), d.t1, C, &d.t2);
            if (!e) e = attn_dev_back(g, d.qkv[l], d.S, d.dS, d.t2, d.dq, B, T, H, hd);
            if (!e) e = linear_back(d.a1[l], C, P(l, QKVW), Gd(l, QKVW), Gd(l, QKVB), d.dq, C3, &d.t1);
            if (!e) e = layernorm_back(d.t1, d.x[l], d.st1[l], P(l, LN1G), Gd(l, LN1G), Gd(l, LN1B), d.dx);
            if (!e) e = Args(g, g.kadd).m(d.t3).m(d.dx).i(NC).flat(NC);
            std::swap(d.dy, d.dx);
        }
    }

    // results
    std::vector<float> rows((size_t)N * 2);
    if (!e) e = cl.EnqueueReadBuffer(g.q, d.rows.m, 1, 0, rows.size() * 4, rows.data(), 0, nullptr, nullptr);
    if (e) return e;
    double total = 0;
    int ok = 0;
    for (int r = 0; r < N; ++r) {
        total += (j.w ? j.w[r] : 1.0) * rows[2 * r];
        ok += rows[2 * r + 1] > 0.5f;
    }
    *loss = (float)(total / wsum);
    if (correct) *correct = ok;
    if (!j.train) return 0;
    for (size_t i = 2; i < np && !e; ++i)
        if (ps[i]->trainable) e = download(g, d.g[i], ps[i]->g.data(), ps[i]->g.size(), true);
    // the embedding gradient is a scatter by token: cheap on the CPU
    Param* tok = ps[0];
    Param* pos = ps[1];
    if (!e && tok->trainable) {
        std::vector<float> dx0((size_t)N * C);
        e = download(g, d.dy, dx0.data(), dx0.size(), false);
        for (int n = 0; n < N && !e; ++n) {
            const float* src = dx0.data() + (size_t)n * C;
            float* a = tok->g.row(j.ids[n]);
            float* q = pos->g.row(n % T);
            for (int c = 0; c < C; ++c) {
                a[c] += src[c];
                q[c] += src[c];
            }
        }
    }
    return e;
}

std::atomic<bool> g_on{false};

void turn_off() {
    g_on = false;
    set_kernels(nullptr);
}

// Large products go to the GPU when it is free; anything else, or an error, uses the CPU.
bool offload(int v, const float* A, const float* B, float* C, int x, int y, int z) {
    Gpu& g = G();
    if (!g_on.load(std::memory_order_relaxed) || (double)x * y * z < g.min_work) return false;
    std::unique_lock<std::mutex> lk(g.mu, std::try_to_lock);
    if (!lk.owns_lock()) return false;
    cl_int e = run(g, v, A, B, C, x, y, z);
    if (e == 0) return true;
    g.cl.Finish(g.q);
    if (!is_memory_error(e)) {
        lk.unlock();
        turn_off();
    }
    return false;
}

// Attention that does not fit or meets an error falls back to the CPU (which also
// recomputes the weights for the backward pass).
bool attn_job(bool backward, const float* qkv, const float* d, float* res, int B, int T, int H, int hd) {
    Gpu& g = G();
    if (!g_on.load(std::memory_order_relaxed) || !g.attn_ok || (double)B * H * T * T * hd < g.attn_min) return false;
    std::unique_lock<std::mutex> lk(g.mu, std::try_to_lock);
    if (!lk.owns_lock()) return false;
    cl_int e = attn_host(g, backward, qkv, d, res, B, T, H, hd);
    if (e == 0) return true;
    g.cl.Finish(g.q);
    if (!is_memory_error(e)) {
        lk.unlock();
        turn_off();
    }
    return false;
}

bool g_attn_fwd(const float* qkv, float* out, int B, int T, int H, int hd) {
    return attn_job(false, qkv, nullptr, out, B, T, H, hd);
}
bool g_attn_bwd(const float* qkv, const float* dout, float* dqkv, int B, int T, int H, int hd) {
    return attn_job(true, qkv, dout, dqkv, B, T, H, hd);
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

// Plain reference attention (double precision) for the correctness check.
void ref_attention(const std::vector<float>& qkv, const std::vector<float>& dout, std::vector<double>& out,
                   std::vector<double>& dqkv, int B, int T, int H, int hd) {
    const int C = H * hd, C3 = 3 * C;
    out.assign((size_t)B * T * C, 0.0);
    dqkv.assign((size_t)B * T * C3, 0.0);
    double scale = 1.0 / std::sqrt((double)hd);
    std::vector<double> p(T), dp(T);
    auto at = [&](int b, int t, int part, int h, int c) { return (double)qkv[((size_t)b * T + t) * C3 + part * C + h * hd + c]; };
    auto dq = [&](int b, int t, int part, int h, int c) -> double& { return dqkv[((size_t)b * T + t) * C3 + part * C + h * hd + c]; };
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h)
            for (int i = 0; i < T; ++i) {
                double mx = -1e300, sum = 0;
                for (int j = 0; j <= i; ++j) {
                    double s = 0;
                    for (int c = 0; c < hd; ++c) s += at(b, i, 0, h, c) * at(b, j, 1, h, c);
                    p[j] = s * scale;
                    mx = std::max(mx, p[j]);
                }
                for (int j = 0; j <= i; ++j) sum += (p[j] = std::exp(p[j] - mx));
                for (int j = 0; j <= i; ++j) p[j] /= sum;
                const float* gr = dout.data() + ((size_t)b * T + i) * C + h * hd;
                double dot = 0;
                for (int j = 0; j <= i; ++j) {
                    double s = 0;
                    for (int c = 0; c < hd; ++c) {
                        out[((size_t)b * T + i) * C + h * hd + c] += p[j] * at(b, j, 2, h, c);
                        s += gr[c] * at(b, j, 2, h, c);
                        dq(b, j, 2, h, c) += p[j] * gr[c];
                    }
                    dp[j] = s;
                    dot += p[j] * s;
                }
                for (int j = 0; j <= i; ++j) {
                    double ds = p[j] * (dp[j] - dot) * scale;
                    for (int c = 0; c < hd; ++c) {
                        dq(b, i, 0, h, c) += ds * at(b, j, 1, h, c);
                        dq(b, j, 1, h, c) += ds * at(b, i, 0, h, c);
                    }
                }
            }
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

bool gpu_attention(bool backward, const float* qkv, const float* dout, float* res, int B, int T, int H, int hd, std::string* err) {
    Gpu& g = G();
    std::lock_guard<std::mutex> lk(g.mu);
    init(g);
    if (!g.ready) {
        if (err) *err = g.info.error;
        return false;
    }
    cl_int e = attn_host(g, backward, qkv, dout, res, B, T, H, hd);
    if (e) {
        g.cl.Finish(g.q);
        if (err) *err = is_memory_error(e) ? "not enough GPU memory" : "OpenCL error " + std::to_string(e);
    }
    return e == 0;
}

double gpu_attention_check(int B, int T, int H, int hd, std::string* err) {
    const int C = H * hd;
    unsigned s = 777;
    auto rnd = [&]() { s = s * 1103515245u + 12345u; return (float)((s >> 9) & 0xFFFF) / 32768.f - 1.f; };
    std::vector<float> qkv((size_t)B * T * 3 * C), dout((size_t)B * T * C), out(dout.size()), dqkv(qkv.size());
    for (float& f : qkv) f = rnd() * 2.f;
    for (float& f : dout) f = rnd();
    if (!gpu_attention(false, qkv.data(), nullptr, out.data(), B, T, H, hd, err) ||
        !gpu_attention(true, qkv.data(), dout.data(), dqkv.data(), B, T, H, hd, err))
        return -1;
    std::vector<double> ro, rd;
    ref_attention(qkv, dout, ro, rd, B, T, H, hd);
    double worst = 0;
    for (size_t i = 0; i < out.size(); ++i) worst = std::max(worst, std::fabs(out[i] - ro[i]) / (1.0 + std::fabs(ro[i])));
    for (size_t i = 0; i < dqkv.size(); ++i) worst = std::max(worst, std::fabs(dqkv[i] - rd[i]) / (1.0 + std::fabs(rd[i])));
    return worst;
}

bool gpu_gpt_step(const GpuGpt& m, const int* ids, const int* tgt, const float* w, int B, int T, bool train, float dropout,
                  uint64_t seed, float scale, float* loss, int* correct, std::string* err) {
    Gpu& g = G();
    if (!g_on.load(std::memory_order_relaxed)) {
        if (err) *err = "the GPU is not turned on";
        return false;
    }
    if ((int)m.params.size() != 2 + 12 * m.nlayer + 4 || m.dim % m.heads != 0) return false;
    std::unique_lock<std::mutex> lk(g.mu, std::try_to_lock);
    if (!lk.owns_lock()) {
        if (err) *err = "the GPU is busy";
        return false;
    }
    if (gpt_floats(m, B, T) * 4 > (double)g.mem * 0.85) {
        if (err) *err = "the model does not fit in GPU memory";
        return false;
    }
    GptJob j{&m, ids, tgt, w, B, T, train, dropout, seed, scale};
    cl_int e = gpt_run(g, j, loss, correct);
    if (e == 0) return true;
    g.cl.Finish(g.q);
    if (err) *err = is_memory_error(e) ? "not enough GPU memory" : "OpenCL error " + std::to_string(e);
    // memory may be held by a larger shape: free it and let the next call retry
    GptBufs& d = g.gpt;
    for (auto* v : {&d.w, &d.g, &d.x, &d.a1, &d.st1, &d.qkv, &d.ho, &d.h, &d.a3, &d.st2, &d.a4})
        for (DBuf& b : *v) release(g, b);
    for (DBuf* b : {&d.xf, &d.stf, &d.logits, &d.ids, &d.tgt, &d.wt, &d.rows, &d.ones, &d.s1, &d.s4, &d.S, &d.dS, &d.dy, &d.dx,
                    &d.t1, &d.t2, &d.t3, &d.yx, &d.dq, &d.d4})
        release(g, *b);
    if (!is_memory_error(e)) {
        lk.unlock();
        turn_off();
    }
    return false;
}

bool gpu_active() { return g_on; }

bool gpu_use(bool on, std::string* msg) {
    Gpu& g = G();
    if (!on) {
        turn_off();
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
    double aerr = gpu_attention_check(2, 77, 3, 8, &err);
    g.attn_ok = aerr >= 0 && aerr < 1e-3;
    if (!g.attn_ok) {
        if (msg) *msg = aerr < 0 ? err : "the GPU attention results differ from the CPU";
        return false;
    }
    // From which size are single products faster on the GPU (copies included)?
    // Only layers outside the whole-model path (LoRA, tables) use this.
    using clock = std::chrono::steady_clock;
    g.min_work = 1e300;
    for (int n : {256, 512, 768}) {
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
        if (best(true) < best(false) * 0.8) {
            g.min_work = (double)n * n * n;
            break;
        }
    }
    g.kernels_name = "OpenCL " + in.name + " + " + cpu_kernels().name;
    g.table = cpu_kernels();
    g.table.name = g.kernels_name.c_str();
    g.table.mm = g_mm;
    g.table.mm_ta = g_mm_ta;
    g.table.mm_tb = g_mm_tb;
    g.table.attn_fwd = g_attn_fwd;
    g.table.attn_bwd = g_attn_bwd;
    g_on = true;
    set_kernels(&g.table);
    if (msg) {
        char b[256];
        std::snprintf(b, sizeof b, "%s (%.1f GB): text models train fully on the GPU", in.name.c_str(), in.mem_gb);
        *msg = b;
    }
    return true;
}

}  // namespace kor
