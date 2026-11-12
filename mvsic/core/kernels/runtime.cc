// ISA backends isolate their codec and Eigen inline symbols, but share this
// scheduler TLS so a backend called by a worker reuses the caller's pool.
// Global-dynamic also supports PIE callers and libraries loaded by Python.
#include "parlay/parallel.h"

#ifndef PARLAY_SHARED_SCHEDULER
#error "runtime.cc requires the private Parlay shared-scheduler overlay"
#endif

namespace parlay {
template <>
thread_local __attribute__((visibility("default"), tls_model("global-dynamic")))
scheduler<WorkStealingJob>::workerInfo scheduler<WorkStealingJob>::worker_info{};
}  // namespace parlay
