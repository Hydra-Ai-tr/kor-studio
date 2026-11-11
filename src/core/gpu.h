// Kor Studio - experimental GPU matrix products through OpenCL.
//
// OpenCL is loaded at run time (OpenCL.dll, libOpenCL.so, the macOS framework),
// so the programs start and work without it. When turned on, the large matrix
// products of training go to the GPU and everything else stays on the CPU; any
// GPU error switches back to the CPU for the rest of the run.
#pragma once
#include <string>

namespace kor {

struct GpuInfo {
    bool found = false;
    std::string name, vendor, platform;
    double mem_gb = 0;
    std::string error;  // why no GPU is usable
};

// Looks for an OpenCL GPU once and remembers the answer.
const GpuInfo& gpu_probe();

// Turns GPU matrix products on (checks the results against the CPU and
// measures from which size the GPU is faster) or off. Returns false and stays
// on the CPU when the GPU cannot help; `msg` says what happened either way.
bool gpu_use(bool on, std::string* msg = nullptr);
bool gpu_active();

// C += op(A) * op(B) on the GPU, with the argument order of Kernels::mm (variant 0),
// mm_ta (1) and mm_tb (2). For tests and benchmarks.
bool gpu_gemm(int variant, const float* A, const float* B, float* C, int x, int y, int z, std::string* err);

}  // namespace kor
