#include <cstddef>
#include "parlay/parallel.h"

#ifndef MVSIC_RUNTIME_TEST_FACTORY
#error "Set MVSIC_RUNTIME_TEST_FACTORY to a unique backend test name"
#endif

extern "C" __attribute__((visibility("default")))
void MVSIC_RUNTIME_TEST_FACTORY(const void** scheduler, size_t* workers, size_t* worker_id) {
  *scheduler = &parlay::internal::get_current_scheduler();
  *workers = parlay::num_workers();
  *worker_id = parlay::worker_id();
}
