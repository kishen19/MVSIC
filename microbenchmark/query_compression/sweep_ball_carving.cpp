#include "mvsic/core/bench_utils.h"
#include "sweep_impl.h"

using namespace mvsic;

template<typename ChPoint, bool metric>
void run(commandLine& P) {
  query_compression_sweep::run_sweep<ChPoint, metric>(
      P, SearchParams::QueryCompression::Carve, "Ball carving");
}

PARSE_DIST_FUNC_AND_RUN(
    run,
    "Query-compression τ sweep (ball carving / MUVERA) using IndexFlat + compressed search.\n\n"
    "Required:\n"
    "  -i <points.pcs>  -q <queries.pcs>  -gt <gt>\n"
    "Optional:\n"
    "  -mm   -k <K> (10)   -o <append.csv>\n"
    "  τ grid: -tau_grid \"v1,v2,...\"  OR  -tau_min -tau_max -tau_steps (default 20)\n"
    "  -tau_spacing linear|geom (default geom: power-law toward τ≈1 for IP [1,0])\n"
    "  -tau_geom_gamma <g> (default 3; larger ⇒ denser near τ≈1 for IP)\n"
    "  -query_subsample <N> (default 1000; cap on queries; <=0 disables)\n"
    "  -query_subsample_seed <S> (default 42)\n"
    "  -dist_func IP|L2\n")
