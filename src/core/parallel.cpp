#include "parallel.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define KOR_PAUSE() _mm_pause()
#else
#define KOR_PAUSE() std::this_thread::yield()
#endif

namespace kor {
namespace {

// Workers wait this long for the next job before they go to sleep, so that
// back-to-back parallel loops do not pay for waking threads up every time.
constexpr auto kSpin = std::chrono::microseconds(200);

struct Pool {
    std::vector<std::thread> workers;
    std::mutex mu;
    std::condition_variable cv, done_cv;
    const std::function<void(int, int)>* fn = nullptr;
    int n = 0, chunk = 1;
    std::atomic<int> next{0};
    std::atomic<int> active{0}, sleeping{0};
    std::atomic<uint64_t> gen{0};
    std::atomic<bool> quit{false};

    explicit Pool(int n_workers) {
        for (int i = 0; i < n_workers; ++i) workers.emplace_back([this] { worker(); });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        cv.notify_all();
        for (auto& t : workers) t.join();
    }
    void run_chunks() {
        for (;;) {
            int b = next.fetch_add(chunk);
            if (b >= n) break;
            (*fn)(b, std::min(n, b + chunk));
        }
    }
    void worker();
    void run(int n_, const std::function<void(int, int)>& f, int chunk_) {
        fn = &f;
        n = n_;
        chunk = chunk_;
        next = 0;
        active = (int)workers.size();
        gen.fetch_add(1);
        if (sleeping.load() > 0) {
            std::lock_guard<std::mutex> lk(mu);
            cv.notify_all();
        }
        run_chunks();
        auto t0 = std::chrono::steady_clock::now();
        while (active.load() != 0) {
            KOR_PAUSE();
            if (std::chrono::steady_clock::now() - t0 > kSpin) {
                std::unique_lock<std::mutex> lk(mu);
                done_cv.wait(lk, [this] { return active.load() == 0; });
                break;
            }
        }
    }
};

thread_local bool tl_in_worker = false;

void Pool::worker() {
    tl_in_worker = true;
    uint64_t seen = 0;
    for (;;) {
        auto t0 = std::chrono::steady_clock::now();
        while (gen.load() == seen && !quit.load()) {
            KOR_PAUSE();
            if (std::chrono::steady_clock::now() - t0 > kSpin) {
                std::unique_lock<std::mutex> lk(mu);
                sleeping.fetch_add(1);
                cv.wait(lk, [&] { return quit.load() || gen.load() != seen; });
                sleeping.fetch_sub(1);
                break;
            }
        }
        if (quit.load()) return;
        seen = gen.load();
        run_chunks();
        if (active.fetch_sub(1) == 1) {
            std::lock_guard<std::mutex> lk(mu);
            done_cv.notify_one();
        }
    }
}

std::mutex g_run_mu;            // serializes users of the pool
std::unique_ptr<Pool> g_pool;
int g_threads = 1;

}  // namespace

void set_threads(int n) {
    n = std::max(1, std::min(n, 256));
    std::lock_guard<std::mutex> lk(g_run_mu);
    if (n == g_threads && (g_pool || n == 1)) return;
    g_pool.reset();
    g_threads = n;
    if (n > 1) g_pool.reset(new Pool(n - 1));
}

int get_threads() { return g_threads; }

void parallel_for(int n, const std::function<void(int, int)>& fn, int min_chunk) {
    if (n <= 0) return;
    if (tl_in_worker || n <= min_chunk) { fn(0, n); return; }
    std::unique_lock<std::mutex> lk(g_run_mu, std::try_to_lock);
    if (!lk.owns_lock() || !g_pool) { fn(0, n); return; }
    int chunk = std::max(min_chunk, (n + g_threads * 4 - 1) / (g_threads * 4));
    g_pool->run(n, fn, chunk);
}

}  // namespace kor
