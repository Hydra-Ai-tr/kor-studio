#include "parallel.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace kor {
namespace {

struct Pool {
    std::vector<std::thread> workers;
    std::mutex mu;
    std::condition_variable cv, done_cv;
    const std::function<void(int, int)>* fn = nullptr;
    int n = 0, chunk = 1;
    std::atomic<int> next{0};
    int active = 0;
    uint64_t gen = 0;
    bool quit = false;

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
        {
            std::lock_guard<std::mutex> lk(mu);
            fn = &f; n = n_; chunk = chunk_;
            next = 0;
            active = (int)workers.size();
            ++gen;
        }
        cv.notify_all();
        run_chunks();
        std::unique_lock<std::mutex> lk(mu);
        done_cv.wait(lk, [this] { return active == 0; });
    }
};

thread_local bool tl_in_worker = false;

void Pool::worker() {
    tl_in_worker = true;
    uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return quit || gen != seen; });
            if (quit) return;
            seen = gen;
        }
        run_chunks();
        {
            std::lock_guard<std::mutex> lk(mu);
            if (--active == 0) done_cv.notify_one();
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
