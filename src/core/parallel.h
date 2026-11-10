// Kor Studio - tiny thread pool with a blocking parallel_for.
#pragma once
#include <functional>

namespace kor {

// Sets the number of threads used by parallel_for (including the caller).
void set_threads(int n);
int get_threads();

// Calls fn(begin, end) over disjoint chunks covering [0, n). Blocks until
// done. Nested or concurrent calls degrade to serial execution.
void parallel_for(int n, const std::function<void(int, int)>& fn, int min_chunk = 1);

}  // namespace kor
