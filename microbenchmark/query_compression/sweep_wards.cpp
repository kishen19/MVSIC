#include "mvsic/core/bench_utils.h"
#include "sweep_impl.h"

using namespace mvsic;

template<typename ChPoint, bool metric>
void run(commandLine& P) {
  query_compression_sweep::run_sweep<ChPoint, metric>(P, SearchParams::QueryCompression::Wards,
                                                       "Ward linkage");
}

PARSE_DIST_FUNC_AND_RUN(
    run,
    "Query-compression τ sweep (Ward linkage) using IndexFlat + compressed search.\n\n"
    "Required:\n"
    "  -i <points.pcs>  -q <queries.pcs>  -gt <gt>\n"
    "Optional:\n"
    "  -mm   -k <K> (10)   -o <append.csv>\n"
    "  τ grid: -tau_grid \"v1,v2,...\"  OR  -tau_min -tau_max -tau_steps (default 20)\n"
    "  Ward defaults: τ in Ward linkage units, sweep 0→16 (merge more as τ increases);\n"
    "    use -tau_max to raise ceiling. Ball-carving IP defaults unchanged (1→0 cosine/dot).\n"
    "  -tau_spacing linear|geom (default geom)\n"
    "  -tau_geom_gamma <g> (default 3)\n"
    "  -query_subsample <N> (default 1000; cap on queries; <=0 disables)\n"
    "  -query_subsample_seed <S> (default 42)\n"
    "  -dist_func IP|L2\n")
