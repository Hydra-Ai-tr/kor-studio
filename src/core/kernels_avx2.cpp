// AVX2 + FMA kernels: this file alone is built with -mavx2 -mfma (/arch:AVX2)
// and is only called after a runtime CPU check.
#define KOR_KNS kavx2
#define KOR_KNAME "AVX2+FMA"
#include "kernels_impl.inc"
