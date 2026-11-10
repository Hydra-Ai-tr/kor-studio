#include "kernels.h"
#include "sysinfo.h"

namespace kor {
namespace kbase { extern const Kernels table; }
#ifdef KOR_HAVE_AVX2_TU
namespace kavx2 { extern const Kernels table; }
#endif

const Kernels& K() {
#ifdef KOR_HAVE_AVX2_TU
    static const Kernels& k = cpu_has_avx2() ? kavx2::table : kbase::table;
#else
    static const Kernels& k = kbase::table;
#endif
    return k;
}

}  // namespace kor
