// ISA backends isolate their codec and Eigen inline symbols, but share this
// scheduler TLS and the sleeping-worker wait/notify implementation. Both must
// belong to one runtime when a backend reuses another library's worker pool.
// Global-dynamic also supports PIE callers and libraries loaded by Python.
#include "parlay/parallel.h"

#ifndef PARLAY_SHARED_SCHEDULER
#error "runtime.cc requires the private Parlay shared-scheduler overlay"
#endif

namespace parlay {
template <>
thread_local __attribute__((visibility("default"), tls_model("global-dynamic")))
scheduler<WorkStealingJob>::workerInfo scheduler<WorkStealingJob>::worker_info{};

#if PARLAY_ELASTIC_PARALLELISM
namespace internal {
// Keep the wait and both notifications in one library: standard libraries may
// use private waiter tables that cannot be split across ISA-isolated DSOs.
void shared_scheduler_wait(const std::atomic<size_t>* counter, size_t value) {
  parlay::atomic_wait(counter, value);
}

void shared_scheduler_notify_one(std::atomic<size_t>* counter) {
  parlay::atomic_notify_one(counter);
}

void shared_scheduler_notify_all(std::atomic<size_t>* counter) {
  parlay::atomic_notify_all(counter);
}
}  // namespace internal
#endif
}  // namespace parlay
