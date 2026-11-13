# Refactor and performance plan — November 12, 2026

Runtime speed at comparable recall decides acceptance. Single-query mean and
tail latency take priority over batch throughput. Keep automatic representative
sizing (`s=0`). Fixed-size, fractional and per-level size tuning is closed.
Stage verified changes; do not commit or push.

## Active continuation

The validated checkpoint is a stopping point for staging, not completion of the
optimization task. The user explicitly asked to continue. Current objective is
faster whole builds at comparable Float retrieval quality, with serial latency
preserved. The24.35-second TQ1 build retains its recall deficit.

1. **Complete:** fresh frozen-original FiQA Float reference57.064 s. Common
   serving-runtime full648-query recall at p32 is56.913580% original versus
   57.268519% current Float; shared returned documents have identical score bits.
2. **Individual screens and integrated validation complete:** borrowed rows
   reduce Float build49.101→45.761 s and peak23.29%, exact e174fc tree. Inner
   TQ1 input-code reuse reduces a separate pair28.107→25.867 s, exact dad381;
   21,984,251 vector occurrences reuse root codes with zero new inner encodings.
   The combined five-file implementation and two ownership/order regressions
   pass normal selected-source builds with both compilers/all three backends,
   24 native suite invocations, 12 exact predictions, installed 48 tests plus
   36 subtests per host ISA, and both external consumers. VNNI is static-only.
   Combined installed Float/TQ1 trees are exact e174fc/dad381. Source 65ed65ad
   (348 files plus three packaging READMEs), environment
   `/tmp/mvsic-selected-view-cache-env`, wheel a158c659. Its original-relative
   batch regression (−3.35%) and previous-runtime regression (−1.63%) remain
   archived. Both alignment screens are closed without production changes.
   A current-core probe measured 1.924 ms of serial request construction.
   Restoring flat Parlay requests/scans/sort/grouping now passes the declared
   three-case serving gate: batch median QPS 15,010 versus original 15,005 and
   source65ed 14,283. Serial mean/p95/p99 beat original by 3.92%/2.74%/2.56%,
   at higher recall and exact current output bytes. Mean batch QPS remains
   1.86% lower than original; serial mean is 1.93% slower than source65ed.
   These limits remain explicit. Source78d7c073 adds only mvivf_search.cc to
   the five exact changes. Final Clang/GCC native, prediction, installed API/YAML
   and consumer checks pass. Wheel5a1eff7d replaces only the facade plus RECORD
   in the prior normal wheel; executable/relocation/layout proof passes.
   Environment `/tmp/mvsic-parlay-requests-env`. All six verified files and
   these reports are staged, preserving the separate unaccepted Flat work.
3. **Private policy stopped at the tail latency gate:** one whole-tree policy preserves
   weighted Float legacy10 inner training, uses exact Float first/final outer
   assignments, and refines intermediate TQ1 top8 UNION mapped previous center
   with Float. FiQA build40.925 s and full648 p32 recall57.978395% improve on the
   fresh Float control49.101 s/57.268519%. Same-p32 pooled mean/tails improve,
   but median full-sweep latency is1.20% slower; preserve that adverse result.
   The sole justified p31 quality point reaches3718hits versus Float p32's3711.
   At matched recall, p31 median/mean/p95 improve1.71%/1.60%/.86%, but p99
   worsens2.93%. The strict gate fails: no batch, repeat, alternate probe or
   public/default promotion. Retain the useful Float-quality build result.
4. **Checkpoint staged:** all 348 selected source blobs plus three packaging
   READMEs match compiled source78d7. The separate unaccepted Flat changes stay
   unstaged. No commit or push. Continue research from this verified checkpoint.

Independent source preparation/compilation may overlap (at most four compiler
jobs); controlled performance runs stay sequential. Stop a failed setting,
not the research objective. Record negative findings briefly and move on.
MV-Vamana connectivity repair and a larger search budget remain later ideas;
the tested R8/L16 warm configuration did not pass quality/cost gates. No broad
parameter grid or fixed representative-size tuning is queued.

## Current checkpoint and preceding accepted source

Current staged/compiled/validated selected source is `78d7c073` (348 source files plus
three packaging READMEs), including exact row views, root TQ1 input reuse and
flat Parlay request grouping. Its installed environment is
`/tmp/mvsic-parlay-requests-env`; final validation is
`parlay-requests-final-validation-01`. Its staged-index audit verifies every
source/packaging blob and excludes unaccepted Flat. Runtime acceptance remains
scoped to the recorded shared-host block,
including mean-batch and source65ed-relative latency tradeoffs above.

The preceding staged source had inventory SHA `89ee99ff`: the validated
`52d34ba4` checkpoint plus the exact TQ1 query-word load change. The runtime files
and kernel match tested source `3028d477`; CMake retains the f1 MVIVF build plus
the scheduler-test timeout. Its six-suite build
has no dependency on the separate unstaged Flat port. See the source audit in
`/ssd2/kishen/mvsic_new/refactor/staged-queryword-checkpoint-audit-01`.

The `04b9a413` worktree build includes Flat and passes seven native suites on
AVX2/AVX-512 under Clang/GCC, twelve saved-tree prediction-byte cases, installed
63 tests plus 42 subtests per ISA, original YAML workflows and the external C++
consumer. A separately hashed packaging supplement restores the benchmark
README without changing any code or binary payload. Installation:
`/tmp/mvsic-runtime-checkpoint-env`. The latest incremental backend/package
validation is `tq1-queryword-validation-01`, with the same native, prediction,
installed and consumer checks. Its wheel replaces only three rebuilt ISA
libraries plus RECORD; every other payload remains identical. Current
installation: `/tmp/mvsic-queryword-env`.

The shared core now owns both Parlay scheduler TLS and sleeping-worker
wait/notify operations. ISA libraries retain private codec/Eigen bindings.
The new idle-pool regression passes with the fix and fails against the old
runtime under both compilers. Assignment/update timers are disjoint.

On ArguAna at 192 workers, the fresh default Float build is 6.890 versus 6.387 seconds,
with peak RSS 3,570 versus 3,021 MiB. Fresh-own-model p32 serial mean/p95/p99 and
recall gates pass, followed by higher batch throughput. Shared-tree serving
also passes. Current output/model bytes are stable across the recorded worker
counts and execution modes. Original padding/reseed defects and lower current
p16 recall prevent claims of quality dominance at every budget. These are
bounded shared-host comparisons. Details are in
[CHECKPOINT_REPORT.md](tests/refactor/CHECKPOINT_REPORT.md).

Public Index/IndexMVIVF and MVClustering use IP/MaxSim. Public Metric and
clustering L2 selection are retired; private Euclidean codebook math remains.
Preserve float centroid updates, weights, norms, optimized batch traversal,
root many-to-many scoring, quantized centers/leaves and TQ8 reranking. Keep
one runtime worker for serial latency. AVX2/AVX-512 are hardware-tested; VNNI
is compiled/inspected only, and AMX/GPU implementations remain future work.

## Next work, in execution order

1. **Final root assignment screen — passed.** The private float-inner policy's
   final root assignment accounts for 13.89M of 25.15M exact float cloud scores.
   This is a work count, not a timing fraction. Screen one fixed TQ1 top16 plus
   float-refinement policy on 4,096 unique documents against the production
   Float C5 state. Prove the published root retains the complete pre-pruning
   center pool before using it. Preserve norms, ties and center positions.
   Initial assignments and inner Lloyd updates remain Float; final rows are
   cold, without invented previous labels. Stop below 99.9% float-winner
   coverage, above1e-5 mean float regret, or below1.25x total speedup including
   encoding/panels/refinement. No shortlist-size grid or whole build before
   this gate. Prepare under `/tmp/mvsic_final_root_float16`; results under
   `final-root-float16-screen-01`.
   The complete241-center C5 pool is proven unpruned. R16 recovers4094/4096
   Float labels; mean regret9.837e-6 barely passes1e-5. Speed is2.251x cached,
   1.589x including model/query encoding; workspace/input/teardown are excluded.
   Do not call two misses negligible or infer whole-build/recall gains.
2. **One root-final-only build — complete, not promoted.** Keep AUTO representatives,
   original samples/weights/seed and Float legacy10 inner updates. A single
   whole FiQA build and all648-query quality check must precede serial mean,
   p95 and p99 comparisons at comparable recall. Batch follows only if serial
   passes. Do not spend the previous policy's extra recall without measurement
   or promote a default from fixed-state assignment agreement. Training uses
   no custom assignment/inner callbacks; only the final full-root partition
   gets TQ1+Float16. Nonroot and standalone clustering stay Float. Capture the
   pre-final root centers to verify byte identity with the Float reference.
   The single build takes44.771 s, including capture I/O, with11.154GB peak
   RSS; it is not a paired speedup measurement. Actual C5 center bytes are
   identical to Float, and only37/57,638 final root labels change. All648-query
   p32 hits improve3711→3729, with exact shared-document score bits. However,
   the balanced eight-sweep serial gate fails: median full sweep+1.23%,
   mean+0.94%, p95+1.52%, p99+2.23%. All four local pairs are slower. No batch,
   alternate probe, retry or promotion. Evidence: `root-final-float16-build-01`.
3. **Finer outer-assignment screen — complete, not promoted.** Use TQ8 top8 with Float
   refinement and mapped genuine previous centers, Float inner10 and exact
   Float C0/final/small-k paths. Existing TQ4/TQ8 scoring both use decoded
   byte panels and int8 queries; TQ8 avoids the moving-center nibble decode.
   Earlier coupled inner/outer experiments do not settle this case. Explicitly
   disable build_with_8btq AND leave inner_function empty so the pilot cannot
   silently quantize inner training. Prepare per-document queries only, omitting
   the unused 1.024GB inner-vector cache. Require all Float winners/score bits
   on the existing C1 sample and at least1.25x total cached-state speedup before
   a whole build. No codec grid. Root graph repair stays later: connectivity
   alone does not explain its reachable-winner misses.
   Actual C1 labels and Float score bits match4096/4096, with zero Float regret.
   Cached median time is1.035914 s versus Float.714573 s:45% slower, mainly
   coarse scoring/query packing. Cold preparation adds.101613 s. The cost gate
   fails; no whole build, retry, codec or shortlist grid. Evidence:
   `outer-tq8-float8-c1-screen-01`.
4. **Exact Float batching screen — complete, not promoted.** Existing Float top1 assignment
   computes an unused query block size and multiplies one document at a time,
   repeating center-matrix packing. Assess the existing blocked AllPairs path
   as a correctness control, then one streaming top1 block of about512 query
   vectors with the unchanged center block size. Preserve per-document min/mean,
   center order, ties, norms and empty handling. Require exact saved labels AND
   score bits plus at least1.10x total assignment speed before any whole build.
   The first fixture gate stops on two score-bit differences, with only the
   first classified as signed zero in that report. A narrow second revision
   retains the historical single-center kernel and classifies all control
   differences. It passes all12 fixtures and4096 C1 winner labels/score bits,
   but takes.831355 s versus Float.713752 s (16.48% slower). All three local
   pairs are slower. Existing AllPairs+argmin improves only4.35%, below10%.
   No block-size grid, whole build or production edit. Evidence:
   `exact-batched-float-c1-screen-01` and `exact-batched-float-c1-screen-02`.
   This branch is closed. Inner distances cannot safely transfer across moved
   representatives; quiet inner training already omits unused final assignment.
5. **Integrate only a passing improvement.** Check exact ownership/sampling,
   both compilers and supported host ISAs, installed Python/YAML/C++ consumers,
   and source/index equivalence. Stage the concrete verified result without
   committing or pushing. Retain rejected variants privately and document them
   briefly; preserve the current staged checkpoint throughout.
6. **Active method migration:** SVHIVF, SVHGraph, MV-Vamana, MUVERA, MPool, then finish Flat's
   pending performance acceptance. Use each original as the reference, with
   fixed-index output/score checks before fresh-build recall and latency, then
   batch. Keep EMDIVF separate. No method is accepted merely because it ports
   or compiles.
   Start raw SVHIVF with explicit raw centers, one frozen original v3 skeleton
   and original typed reference. Original oracle and raw concrete port are
   prepared separately under `/tmp`; neither is a production replacement yet.
   Prove fixed-topology IDs/score bits, candidate aggregation and original
   reranking semantics before fresh builds or latency. Preserve the current
   staged MVIVF selection and do not absorb the unaccepted Flat integration.

## Closed screens and constraints

- Public retrieval/MVClustering Metric and L2 selection are retired. Private
  Euclidean quantizer codebook math remains. L2/PrefixDoubling, tested small
  inner ANN, e-means/EMA variants, fixed/per-level representative sizes and
  leaf duplicate compaction are closed on current evidence.
- Warm previous-center starts help outer ANN, but the tested R8/L16 policy
  fails quality and total-cost gates. Do not repeat it or start a graph grid.
- The float-inner/TQ1+Float8 whole-tree policy improves FiQA build/quality but
  fails matched-recall p99. Its extra routing/leaf work is measured; it is
  not a new default. Panel alignment and function alignment64 are also closed.
- FHT is already used. Root query encodings are reused; exact inner TQ1 reuse
  is now staged. The current root profile is99.09% scoring within its matching
  assignment envelopes, not whole-build CPU time. Overlapping inner phase
  walls cannot be added or called CPU shares. Hardware perf sampling is denied
  by host policy; do not retry it.
- Exact row views and Parlay request grouping are staged and validated. Their
  earlier confounded pre-wakeup measurements are historical, not pending work.
  Retain all adverse batch means and the latest block's tiny median margin.

Detailed experiments and failures remain in
[TRAINING_EXPLORATION.md](tests/refactor/TRAINING_EXPLORATION.md) and the SSD
artifacts. No large equal-recall quantized-training win has been established.

## Later method migration

Keep method-specific folders under `mvsic/algorithms`. Shared numeric/data
facilities belong in meaningful core subfolders; shared native benchmark code
stays under algorithms. Continue one method at a time: SVH, MV-Vamana, MUVERA,
MPool and Flat, preserving original feature/neighbor bytes and testing serial
latency before batch. The Flat port is applied and correctness-tested but
remains unstaged after its separate installed latency failure. MPool is
preparation-only. Ignore EMDIVF, which is being developed independently.

Retain original implementations until their replacements are verified. Remove
obsolete method-local files when that replacement is accepted; avoid a broad
cleanup of methods still awaiting migration. Keep the exact installable YAML
benchmark interface and method-specific statistics semantics.

## Execution constraints

Read the EMDIVF marker before heavy work:
`/home/kishen/MVSIC/emdivf_experiments/artifacts/EMDIVF_BENCHMARK_ACTIVE.json`.
The original repository remains read-only. Use 192 unpinned Parlay workers for
necessary parallel experiments and one ancillary library worker; no 192-vs-96
grid or load-guard waiting. Serial latency uses one worker. Bound compilation
parallelism to four jobs and run performance work separately. Place generated
results under `/ssd2/kishen/mvsic_new/refactor`, preserve failed/preflight attempts,
and record source, inputs, actual loaded libraries and every timing sample.
