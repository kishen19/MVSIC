#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
#include "parlay/parallel.h"

using TestBackend = void (*)(const void**, size_t*, size_t*, void (*)(void*), void*);
extern "C" void mvsic_runtime_test_one(const void**, size_t*, size_t*, void (*)(void*), void*);
extern "C" void mvsic_runtime_test_two(const void**, size_t*, size_t*, void (*)(void*), void*);

struct ActiveWorkers {
  std::atomic<size_t> active{0}, peak{0}, callbacks{0};
};

void finite_work(void* context) {
  auto& workers = *static_cast<ActiveWorkers*>(context);
  const size_t active = workers.active.fetch_add(1) + 1;
  size_t peak = workers.peak.load();
  while (peak < active && !workers.peak.compare_exchange_weak(peak, active)) {}
  // Finite work exposes participation without a barrier that could deadlock
  // when notifications are broken and the caller must finish every job itself.
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  workers.callbacks.fetch_add(1);
  workers.active.fetch_sub(1);
}

void verify_idle_workers(TestBackend backend, const void* expected,
                         std::atomic<bool>& correct) {
  // The scheduler normally parks idle workers after 10ms. Exercise wakeup after
  // parking, which the scheduler-address and already-active-pool checks miss.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  ActiveWorkers active;
  const void* actual = nullptr;
  size_t workers = 0, id = 0;
  if (backend) {
    backend(&actual, &workers, &id, finite_work, &active);
  } else {
    actual = &parlay::internal::get_current_scheduler();
    workers = parlay::num_workers();
    id = parlay::worker_id();
    parlay::parallel_for(0, 32, [&](size_t) { finite_work(&active); }, 1);
  }
  if (actual != expected || workers != 4 || id != 0 || active.peak.load() < 2
      || active.callbacks.load() != 32) {
    std::cerr << "Idle pool did not wake across a library boundary: "
              << active.peak.load() << " concurrent callbacks\n";
    correct.store(false);
  }
}

void verify_one_worker() {
  parlay::execute_with_scheduler(1, [&] {
    for (long granularity : {0, 1, 17, -1})
      for (bool conservative : {false, true}) {
        std::vector<size_t> order;
        parlay::parallel_for(0, 257, [&](size_t i) {
          order.push_back(i);
          size_t nested = 0;
          parlay::parallel_for(0, 3, [&](size_t j) {
            if (j != nested++) throw std::runtime_error("One-worker nested loop reordered");
          }, granularity, conservative);
        }, granularity, conservative);
        if (order.size() != 257) throw std::runtime_error("One-worker loop lost callbacks");
        for (size_t i = 0; i < order.size(); ++i)
          if (order[i] != i) throw std::runtime_error("One-worker loop reordered callbacks");
      }
    for (bool conservative : {false, true}) {
      size_t step = 0;
      parlay::par_do([&] { if (step++ != 0) throw std::runtime_error("Left branch reordered"); },
                     [&] { if (step++ != 1) throw std::runtime_error("Right branch reordered"); },
                     conservative);
      for (size_t failure : {0, 1, 2}) {
        size_t visited = 0;
        bool caught = false;
        try {
          parlay::parallel_for(0, 257, [&](size_t i) {
            ++visited;
            if (i == failure) throw std::runtime_error("expected loop error");
          }, 1, conservative);
        } catch (const std::runtime_error&) { caught = true; }
        if (!caught || visited != failure + 1)
          throw std::runtime_error("One-worker loop exception did not stop at the failing row");
      }
      for (bool fail_left : {false, true}) {
        bool left = false, right = false, caught = false;
        try {
          parlay::par_do([&] {
            left = true;
            if (fail_left) throw std::runtime_error("expected left error");
          }, [&] {
            right = true;
            throw std::runtime_error("expected right error");
          }, conservative);
        } catch (const std::runtime_error&) { caught = true; }
        if (!caught || !left || right == fail_left)
          throw std::runtime_error("One-worker fork exception changed branch order");
      }
    }
  });
}

int main() {
  // A broken TLS boundary should fail with a small unintended pool, rather
  // than creating hardware_concurrency() workers in each backend and caller.
  setenv("PARLAY_NUM_THREADS", "4", 1);
  verify_one_worker();
  std::atomic<bool> correct{true};
  for (unsigned count : {1, 3})
    parlay::execute_with_scheduler(count, [&] {
      const void* expected = &parlay::internal::get_current_scheduler();
      parlay::parallel_for(0, 300, [&](size_t) {
        const size_t expected_worker = parlay::worker_id();
        for (auto backend : {mvsic_runtime_test_one, mvsic_runtime_test_two}) {
          const void* actual = nullptr;
          size_t workers = 0, id = 0;
          backend(&actual, &workers, &id, nullptr, nullptr);
          if (actual != expected || workers != count || id != expected_worker) correct.store(false);
        }
      }, 1);
    });
  // A caller-owned pool must wake when either isolated backend publishes work.
  parlay::execute_with_scheduler(4, [&] {
    const void* expected = &parlay::internal::get_current_scheduler();
    verify_idle_workers(mvsic_runtime_test_one, expected, correct);
    verify_idle_workers(mvsic_runtime_test_two, expected, correct);
  });
  // Create the default pool in backend one, then publish from backend two and
  // the caller. Its worker sleep code belongs to a different DSO in both cases.
  const void* expected = nullptr;
  size_t workers = 0, id = 0;
  mvsic_runtime_test_one(&expected, &workers, &id, nullptr, nullptr);
  verify_idle_workers(mvsic_runtime_test_two, expected, correct);
  verify_idle_workers(nullptr, expected, correct);
  verify_idle_workers(mvsic_runtime_test_one, expected, correct);
  if (!correct.load()) {
    std::cerr << "Parlay scheduler sharing or sleeping-worker wakeup failed\n";
    return 1;
  }
  std::cout << "Shared Parlay scheduler checks passed\n";
  return 0;
}
