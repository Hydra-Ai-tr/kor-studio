#include "kernels.h"
#include "sysinfo.h"

#include <atomic>

namespace kor {
namespace kbase { extern const Kernels table; }
#ifdef KOR_HAVE_AVX2_TU
namespace kavx2 { extern const Kernels table; }
#endif

const Kernels& cpu_kernels() {
#ifdef KOR_HAVE_AVX2_TU
    static const Kernels& k = cpu_has_avx2() ? kavx2::table : kbase::table;
#else
    static const Kernels& k = kbase::table;
#endif
    return k;
}

static std::atomic<const Kernels*> g_override{nullptr};

void set_kernels(const Kernels* k) { g_override.store(k, std::memory_order_release); }

const Kernels& K() {
    const Kernels* k = g_override.load(std::memory_order_acquire);
    return k ? *k : cpu_kernels();
}

}  // namespace kor
