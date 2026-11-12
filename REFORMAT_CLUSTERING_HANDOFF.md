# MVClustering and CPU kernels handoff — compiled and correctness checked

Work tree: `/home/kishen/MVSIC-new/MVSIC`, branch `reformat`. This replaces the
initial pre-compilation handoff. Root owns final staging and acceptance. No
commit or push has been made. Performance acceptance and experimental default
selection are still pending; passing correctness tests is not a speed claim.

## Coordination and validation checkpoint

Before heavy work, reread
`/home/kishen/MVSIC/emdivf_experiments/artifacts/EMDIVF_BENCHMARK_ACTIVE.json`.
It was released (`active=false`) at `2026-11-10T23:40:19.530186+00:00` after the
original EMDIVF window. Root coordinates all compilation/benchmarks; a released
marker does not reserve the machine. The original/refactor ArguAna native pilot and ten-recipe float training pilot
have completed. The latter waited for an unrelated overlapping benchmark to
finish, then ran alone. The fresh reference quantization phase was interrupted after a competing
compiler appeared during TQ4 construction. It is incomplete and its timings are
quarantined. The candidate compiled and passed native checks. After interrupted
or refused attempts, its clean eight-case microbenchmark completed in
`inner-assignment-004`. The measured AVX2 scalar regression led to a temporary
gate; the subsequent SIMD implementation passed expanded checks and a second
clean matrix (`inner-assignment-005`). TQ4 packing is now enabled for every
compiled CPU backend. Root completed the final combined build, validation,
package and parity checks; the candidate whole-index quantization phase and
remaining old-baseline TQ4/TQ1 runs have also completed.
The measured TQ1 assignment SIMD prototype has now been integrated into the
clustering source, with independent decoded-codec regression coverage. The
combined Clang/GCC native suites, installed package and strict parity checks
passed. Its fresh whole-index TQ1 build completed in 31.005 seconds versus
131.505 seconds before the optimization, preserving the exact serialized index
and all measured recalls. Source remains frozen; float training stays default.
A later serving-only TQ1 fallback optimization and long-query overflow fix
passed combined Clang/GCC, package and parity checks. The byte-accumulation
optimization remains provisional while the coordinated 44-process whole-query
comparison runs. It does not replace or extend the training measurements below.
The installed `ip-wheel-01` remains unchanged for a clean baseline rerun. Recheck live activity
as well as the marker before any additional heavy work. Generated results belong under `/ssd2/kishen/mvsic_new`.

Current evidence under `/ssd2/kishen/mvsic_new/refactor`:

- `ip-validation-01/{avx2,avx512}`: all five native suites pass after the
  IP-only retrieval change. L2 outer/inner clustering remains supported.
- `ip-wheel-01/tests/stdout.log`: all 10 installed API/CLI tests pass, including
  retained L2 clustering. This installed wheel is the frozen training-pilot
  implementation; all 31 package files were verified against its archive.
- `ip-parity-01/result.json`: all 84 IP retrieval parity checks pass.
- `packed-validation-01/{avx2,avx512}`: all five native suites pass with the
  packed-center candidate. `yaml-wheel-01` passes all 11 installed API/CLI tests
  from outside the checkout, and `cpp-install-02` passes the installed C++
  consumer. The new wheel lives in `/tmp/mvsic-final-env`; the frozen
  `/tmp/mvsic-refactor-env` baseline package remains unchanged.
- `packed-parity-01`: all 84 strict original/refactor IP parity checks pass
  with the packed-center candidate and current installed package sources.
- `final-validation-02` and `gcc-final-validation-01`: all five native suites
  pass on both AVX2 and AVX-512 with TQ1 assignment SIMD and the final serving
  tie fixes. `final-wheel-02` is the measured TQ1 training snapshot;
  `final-parity-02` passes all 84 strict parity checks.
- `arguana-quantization-tq1-simd-001`: one fresh TQ1 whole build and all eight
  single/batch recall points completed cleanly; details and limitations below.
- `final-build-04`, `final-validation-03` and `gcc-final-validation-02`:
  provisional serving byte accumulation and overflow correction compile and
  pass all six native suites on both ISAs, including the same-source AVX-512
  panel test. `final-wheel-03`, `cpp-install-05` and `final-parity-03` pass the
  installed package, installed C++ consumer and all 84 strict parity checks.

Earlier validation before retiring retrieval L2:

- `gcc-validation-01/{avx2,avx512}` and
  `clang-validation-01/{avx2,avx512}`: all five CTest suites pass, including
  MVClustering, panels, dispatch and shared scheduler. The original implementation
  is compared using Clang 15, matching its compiler requirements; the GCC tree is
  a refactor correctness check, not a mixed-compiler performance baseline.
- `wheel-validation-02/tests/stdout.log`: all 10 installed Python/API/CLI tests
  pass after the TQ4 padding correction. `cpp-install-01` checks an installed
  CMake consumer. Python registration cleanup removed all 25 orphan clone
  `register_*.cc` files and `bind_macros.h`; comparison bindings still use files
  from the untouched original checkout.
- `parity-03/result.json`: all 168 same-skeleton checks pass. The correctness
  reference is original AVX2 **single-query** scoring: original AVX-512 TQ4 has
  an IP padding bug, and original AVX2 TQ1 batch returns a placeholder score.
  Refactor AVX2/AVX-512, single/batch, root-M2M, partial visits and IP/L2 oracle
  checks are strict. See `tests/refactor/README.md` for the recorded reference
  exclusions; do not treat them as refactor exceptions or timing results.
- The TQ4 fix masks padding lanes while preserving legitimate zero vectors.
  This also corrects TQ4 clustering assignment; old incorrect center assignments
  are not a valid numerical oracle.

VNNI implementations compile but this host has no usable VNNI/AMX hardware.
Neither is hardware validated. AMX remains a capability/design placeholder.

## Completed clustering implementation

The public non-template facade is in
`mvsic/algorithms/mvivf/mvclustering/mvclustering.h/.cc`. It accepts runtime
`Metric`, returns owned center storage and center-position assignments, and
exposes `assign_point_clouds` for assignment checks. Existing template trainer,
configuration and seeding files moved from `core/mvclustering` to that same
method directory; the live internal trainer is `mvclustering_impl.h`
(Bazel `:mvclustering_impl`). The abandoned `lower_bounds.h` and its commented
BUILD rule were removed. `mvclustering_8bit.h` remains used by Flat/Spill.

`mvclustering_kernels.cc` contains private metric specializations behind the
coarse backend factory. The AVX2, AVX-512 and VNNI DSOs compile these kernels;
the public facade selects one before training. `mvclustering_internal.h`
provides `ClusteringContext`, whose immutable root caches are shared across
recursive MVIVF training, and `validate_clustering_config`, also called by the
index constructor for leaf-only roots.

- `assignment_quantizer` supports None/TQ8/TQ4/TQ1 independently of serving
  quantization, for both outer cloud and inner vector assignment. The legacy
  `build_with_8btq` flag aliases TQ8; conflicting explicit choices are rejected.
- Float centroid updates, document weights, sampled rows and outer metric stay
  intact. Seeded centers have positional IDs even when source IDs repeat.
  TQ1 clustering uses the general-norm single-vector codec and composes cloud
  minimum/mean scores, including zero and nonunit vectors. Unit-only serving
  TQ1 is a separate boundary contract.
- Immutable input encodings, norms and rotations are reused. Changed float
  centers are encoded again after every update. TQ8 root byte/norm caches can
  be shared across different outer/inner metrics through correctly typed
  models; no reinterpret casts or stale centroid distances are used.
- Shape, ownership, configuration, finite norm range, dimension and allocation
  products are checked at boundaries. Quantized dimensions are capped at 65535
  to bound integer panel accumulation. Squared norms are conservatively bounded
  by `FLT_MAX/(16*UINT32_MAX)` so any representable cloud-length score reduction
  stays finite, without adding checks or wider accumulators to hot loops.
- Parlay uninitialized storage is used only for primitive buffers fully written
  before access. Centroid accumulators and fresh quantizer padding remain zero
  initialized. `filter_flattened` constructs each owning row directly instead
  of copying an uninitialized row prototype. Standard vectors retain their
  normal object lifetimes and ownership.

Tests cover float/legacy parity, all codecs and both metrics, weighted updates,
nonunit/zero inputs, arbitrary dimensions, duplicated IDs, sampled/root-mapped
caches, seeded repeatability, ownership, invalid boundaries and overflow cases.
Experimental tests cover cross-metric TQ8 reuse, scalar seeding objective/ties,
adaptive layout/means, zero-weight/zero-vector cases and the final-assignment
optimization. Initial compiler errors from Parlay slice `.data()` were corrected
with pointer-returning `.begin()`; standard-vector offsets still use `.data()`.

## Inner k-means, provenance and experimental controls

Local trainers now live under `mvsic/core/kmeans/`, retaining
`lloyds_kmeans.h` and `kmeans_util.h`. The minimal 12-header Parallel-KMeans
closure is vendored under its useful `lloyds/`, `lloyds_weighted/` and `seeding/`
subdirectories. Original baseline include names resolve against this same
pinned subset, so no external/private KMeans download is needed.

Provenance: `https://github.com/ParAlg/Parallel-KMeans`, revision
`6cdb5b8649ea254c31a329fcdbbf6537e01452b2` (`mvsic` branch). Headers retain the
original bodies apart from provenance comments and the pre-existing `inline`
ODR fix on `symmetrize()`. No root license or per-file notices exist in that
subset; no license was invented. `core/kmeans/README.md` records the user-supplied
paper citation: Xiaojun Dong et al., **UltraKMeans: Fully Scalable and High
Quality k-means**, PPoPP 2027, accepted/to appear; author page
`https://kishen19.github.io/`, no DOI yet.

A separate `/tmp/mvsic-parallel-kmeans-latest` checkout fetched upstream `main`
at `048e1ed6ff956febe80874c30f2178119ef104e4` (June 10 2026), without mutating
original sources. A fresh GitHub fetch on November 11 confirmed this remains
upstream `main`; there are no new commits beyond the previous audit. Selected newer improvements were adapted into the local
trainer with section provenance; the original pinned subset stays unchanged.
The upstream triangle-pruning path was not imported: its competitor/incumbent
comparison can accept a candidate using a loose incumbent upper bound without
evaluating the incumbent.

Defaults remain `inner_metric="same"`, `inner_init="default"`,
`inner_backend="legacy"`. No experimental option was promoted to a default.

| Field | Choices | Scope |
| --- | --- | --- |
| inner_metric | same / l2 / ip | Inner objective only; L2 arithmetic means, IP spherical means |
| inner_init | default / uniform / kmeans++ / prefix_doubling | Explicit seeded initialization; historical default preserved |
| inner_backend | legacy / adaptive | Float storage, tiling and mean reduction implementation |

Explicit IP seeding samples normalized directions with positive chord-distance
weights while actual assignment/update scores retain raw input norms. Zero
vectors remain zero. Weighted D² sampling uses document weights. Float-only
initial assignment reuse records the actual objective to the selected centers,
separately from sampling distances; quantized backends recompute their own
initial assignment. Near ties may round differently between scalar seeding and
Eigen scoring, so tests compare objective tolerance instead of requiring
identical labels at such ties.

The adaptive float backend uses flat storage, approximately 4 MiB Eigen tiles and
streaming float centroid updates, avoiding repeated huge-page allocations for
small inner problems. The historical layout/256-row tiles remain the default.
TQ8 already has flat input and streaming means. Quiet Lloyd training skips the
assignment after the last mean update because only centers are returned; verbose
cost reporting retains it. An instrumented real-backend regression checks that
all earlier update labels and returned centers match the complete schedule.

## CPU/dependency organization

`core/kernels/{capabilities,dispatch,dot_intel}` isolate runtime capability checks,
coarse byte-panel dispatch and small target-attributed Intel functions. AVX2 and
non-VNNI AVX-512 byte dots split low7/high1 components to avoid signed-int16
saturation; VNNI uses non-saturating dpbusd. Existing fused multi-vector panel
kernels remain the serving/training fast paths. Do not describe generic
`batch_dot` as replacing them.

CMake uses `cmake/Dependencies.cmake` plus a private build overlay generated by
`prepare_dependencies.py`; original dependency sources and caches stay readonly.
The patched Parlay shared scheduler has one TLS definition in common core
`core/kernels/runtime.cc`, using global-dynamic TLS so executable and Python
`dlopen` users share a scheduler safely across backend DSOs. Baseline targets use
unpatched Parlay and original source includes. Root owns CMake/packaging details.

## Next controlled experiments

`tests/refactor/build_quality.py` independently builds every recipe and searches
its own tree through the installed config interface. It records source/runtime
hashes, affinity, data/query identity, build/preparation memory and time, raw
search results, identical-budget recall deltas and observed matched-recall
ratios. It gates heavy subprocesses on the coordination marker. It does not
select a winner automatically.

Prepared source configs:

- `configs/clustering_arguana_pilot_base.json` and
  `clustering_arguana_pilot.json`: ten float-assignment recipes, model seed 0,
  eight build/batch workers on node 0 CPUs 0/4/8/12/16/20/24/28, single-query CPU 0,
  128 queries sampled with seed 42, probes 4/8/16/32, rerank 50, Wards 0.5,
  TQ4 centers/TQ1 leaves, TQ8 rerank, root-M2M. Ten independent builds and 80
  operating points; three timing repetitions. No standalone diagnostics or
  extra Python-call timing in this first screen.
- `configs/clustering_base.json` and `clustering_experiment.json`: the same
  ten recipes, eight workers, three model seeds, 1,000 queries, wider grid and
  separate sampled clustering diagnostics: 30 builds and 480 operating points.
- Recipes separate IP uniform, L2 uniform, IP KMeans++, IP PrefixDoubling,
  backend changes and L2 PrefixDoubling/adaptive using one changed field per
  parent-child edge. Outer assignment, serving metric and ground truth remain
  IP even for L2 inner training. Never normalize L2 arithmetic means to make
  those objectives appear equivalent.
- After reviewing float results, explicitly set `selected_training` in a copied
  spec, then run `--phase quantization`. It creates fresh paired float/TQ8/TQ4/TQ1
  builds with identical serving policy. Pilot phase 2 has four builds and 32
  operating points. Root selected `reference` after reviewing all float curves.

Both training plans and every possible quantization follow-up were schema
validated in memory without importing native code or launching an experiment.
See `tests/refactor/clustering_experiment.md` for commands and interpretation.

Launch only in the root-coordinated quiet host window. The completed first output
is `/ssd2/kishen/mvsic_new/refactor/arguana-training-pilot-001` (10 builds and
80 operating points); `pilot_review.json` records full curves and comparisons to
the reference. Run from `/tmp` with the installed environment and
`PYTHONPATH`/`LD_LIBRARY_PATH` unset, keeping `ip-wheel-01` unchanged throughout.
Investigate build, latency, throughput, memory or recall regressions before
acceptance. One seed/128 queries is a screen, not evidence for changing defaults.
For promising recipes, confirm with multiple seeds and a sufficient recall
budget envelope; report missing recall matches, not extrapolated wins. Separate
clean/incremental compilation measurements remain root-coordinated work.

## Measured float pilot and candidate checkpoint

One model seed and 128 fixed queries are an exploratory screen, not evidence
for promoting defaults. Recall below is at 32 probes with TQ8 reranking of 50
candidates; the full review retains all four probe points and both schedules.

| Inner recipe | Build seconds | Peak RSS GiB | Recall@10 |
| --- | ---: | ---: | ---: |
| reference | 38.391 | 1.137 | 0.94688 |
| ip_uniform | 38.391 | 1.124 | 0.94688 |
| l2_uniform | 36.763 | 1.036 | 0.93281 |
| ip_kmeanspp | 52.783 | 1.153 | 0.93750 |
| ip_prefix | 50.640 | 1.107 | 0.94609 |
| ip_adaptive | 37.323 | 1.046 | 0.94531 |
| l2_adaptive | 35.972 | 1.018 | 0.92813 |
| ip_kmeanspp_adaptive | 51.778 | 1.104 | 0.93750 |
| l2_prefix | 44.509 | 1.165 | 0.93281 |
| l2_prefix_adaptive | 43.675 | 0.993 | 0.93281 |

The first two timings only coincide after rounding: 38.390760377049446 versus
38.39111624006182 seconds. Both records have `resumed=false`, independent
processes, different configuration fingerprints and different artifact hashes.
IP-uniform matched reference recall within the declared 0.001 tolerance across
the observed grid; single-query QPS ratios were 1.003–1.025 and batch ratios
1.000–1.103. These small-sample differences do not establish a robust win.
L2 variants used less memory but missed the reference high-recall point in this
grid and required more probes for several middle recall targets. IP KMeans++
improved some middle points but cost more build time and missed the reference
high end. IP PrefixDoubling matched the high end within tolerance but needed
32 probes to match the reference's 16-probe recall and built about 32% slower.
No defaults changed. Root selected `reference` to isolate assignment quantization.

The first float/TQ8/TQ4/TQ1 baseline at
`/ssd2/kishen/mvsic_new/refactor/arguana-quantization-baseline-001` is interrupted.
Its sibling `-spec.json` records explicit selection and the unchanged pilot
base-config hash. `timing_interference.json` records the compiler observation,
clean controller interruption (exit 130), and individual process intervals.
Float and TQ8 completed; TQ4 was terminated and TQ1 was not started. The compiler
exited before its placement could be captured, so contention was possible rather
than proven on specific cores; conservatively exclude this phase from timing
acceptance and rerun it in a fresh directory. `partial_recall_review.json` is
quality-only: float recall at probes 4/8/16/32 was
0.74453/0.84063/0.90234/0.94688; TQ8 was
0.75313/0.84063/0.89375/0.93047. Preserve the installed baseline wheel.

A separate candidate now packs TQ4 inner centers into full panels
instead of one vector per cloud (which wasted 15 of 16 lanes). It reuses the
existing TQ8 `SingleCloudArgmin` through a generic encoded-set parameter.
The shared reduction now chooses the lowest center ID on ties across panels;
previous AVX-512 code could prefer center 16 over equally scoring center 1 by
choosing the lowest lane. Float assignment and updates are unchanged.

Candidate files: `mvclustering_kernels.cc`, the `SingleCloudArgmin` class in
`core/quantization/turboquant_8bit_mv.h`, and
`tests/refactor/inner_assignment_bench.cc`. The latter checks decoded byte/norm
oracles, odd dimensions, query tails, padding, zero vectors, cross-panel ties,
and repeated weighted IP/L2 float updates before alternating unpacked/packed
kernel and encode-plus-assignment timings. CMake comparison targets are
`mvsic_inner_assignment_avx2` and `mvsic_inner_assignment_avx512`. The frozen
candidate compiled in `packed-build-01`; all five CTest suites passed on both
ISAs in `packed-validation-01`, including concurrent first-use TQ8 reranking.
AVX2 packed argmin currently has a scalar fallback; its existing serving panels
have separate AVX2 SIMD. Do not claim SIMD argmin or a speedup without measuring.

`tests/refactor/inner_assignment.py` pins executables and their MVSIC DSOs,
records compiler commands and source provenance, checks the marker throughout,
sets one worker on CPU0, and retains seven timing samples (10 loops per sample)
for each kernel and encode-plus-assignment variant. A host CPU-delta monitor
interrupts only its own process group when unrelated sustained CPU work appears.
Each native process checks its numerical oracles before timing.

Its first run, `inner-assignment-001`, is **incomplete and quarantined**. At
11:50:19/21 UTC an external `stream-baseline` process (PID 3231091, parent
3231059, affinity CPUs 0–63) consumed approximately 56 then 62 cores. The guard
stopped our controller cleanly (exit 130). Three TQ4 cases completed their
embedded checks before the interruption; the remaining cases were rerun below.
Do not use this interrupted run to accept the candidate. Its immutable binaries,
raw timing distributions, process records and `host_samples.jsonl` are retained.

`inner-assignment-002` refused to start because packaging had reconfigured the
build tree without comparison targets, removing their compile-command entries.
`inner-assignment-003` verified the compiler, executable and DSO hashes against
the pinned first-run provenance, then refused its quiet-host preflight when a
new external 64-core job appeared. Neither attempt launched a native case.

`inner-assignment-004` completed **all eight cases with all checks passing**.
Clang 15 Release flags were `-O3 -march=x86-64-v3`, with
`-march=skylake-avx512` for the AVX-512 target. One worker used CPU0 with
2800 input vectors, 140 centers and dimension 128. Each result retains seven
samples of ten loops, alternating variant order. All 21 host-monitor samples
were quiet; maximum unrelated activity was 0.435 cores total and 0.144 cores
for any single process. `review.json` records interpretation and limitations.

Packed versus singleton-center speed ratios (higher is faster):

| ISA | Codec | Objective | Kernel median/min | Encode + assign median/min |
| --- | --- | --- | --- | --- |
| AVX2 scalar | TQ4 | IP | 0.990 / 0.949 | 0.979 / 0.957 |
| AVX2 scalar | TQ4 | L2 | 1.036 / 1.021 | 1.072 / 1.044 |
| AVX-512 | TQ4 | IP | 21.906 / 21.737 | 16.306 / 16.282 |
| AVX-512 | TQ4 | L2 | 22.239 / 22.182 | 16.462 / 16.418 |
| AVX2 scalar | TQ8 | IP | 0.990 / 0.988 | 0.999 / 0.947 |
| AVX2 scalar | TQ8 | L2 | 1.049 / 1.016 | 1.059 / 1.018 |
| AVX-512 | TQ8 | IP | 21.844 / 21.424 | 18.688 / 18.764 |
| AVX-512 | TQ8 | L2 | 22.206 / 22.044 | 19.252 / 18.829 |

The TQ4 AVX-512 encode-plus-assignment medians fell from 19.755 to 1.212 ms
(IP) and 20.413 to 1.240 ms (L2). The AVX2 IP median regressed from 22.686 to
23.170 ms, and its minimum also regressed. Root therefore chose a coarse
compile-time AVX-512-only gate for TQ4 packing in `mvclustering_kernels.cc`;
AVX2 retains its previous singleton-center path for both objectives. This gate
was validated in the subsequent native build. The MVIVF agent then implemented
AVX2 SIMD argmin in the codec header; the next measurement below decides whether
to replace this temporary gate.
Production TQ8 already used packed argmin, so its ratios above are diagnostic
and are **not new refactor gains**. No training defaults changed.

The subsequent AVX2 SIMD candidate compiled in `avx2-packed-build-01`, with all
five native suites passing on AVX2 and AVX-512 in `avx2-packed-validation-01`.
Its expanded oracle adds 8-center half-panel boundaries, center counts
8/9/15/31/32, one/three-query tails, ties between IDs 1/8 and 8/16, and retains
all existing decoded-score, zero/padding, cross-panel and weighted-update checks.

`inner-assignment-005` completed all eight expanded checks and timing cases in
a strictly quiet window, with the same CPU0, sizes, loops and repetition counts.
All 15 host samples were quiet (maximum unrelated total 0.488 cores and single
process 0.158 cores). The new AVX2 TQ4 encode-plus-assignment speedups against
the singleton path measured in the same process were **11.582x IP / 11.529x L2**.
Against the previous packed scalar implementation, actual TQ8 production
encode-plus-assignment medians fell from **23.164 to 1.982 ms for IP (11.686x)**
and **23.237 to 2.039 ms for L2 (11.397x)**. Minimum-time ratios were 11.760x and
11.323x, respectively. TQ8 kernel-only median speedups were 13.289x / 12.640x.
AVX-512 controls remained approximately stable; packed assignment median ratios
were 0.978–0.997 and kernel ratios 0.993–1.004 between the two runs.

`inner-assignment-005/review.json` retains both prior and candidate distributions,
explicitly distinguishing TQ8's real packed-scalar-to-SIMD gain from the
artificial singleton comparison. Based on these checks and measurements, root
approved reenabling AVX2 TQ4 packing. The temporary ISA gate is now removed:
all public CPU backends have AVX2 or better, and both IP/L2 use packed TQ4
assignment. The low-level helper retains its generic scalar fallback. This final
source change is frozen for the coordinated combined build; whole-build/recall
measurements still remain. Float/default training choices are unchanged.

The end-to-end checkpoint below records subsequent build and recall measurements.
Preserve the old installed baseline. TQ4/TQ1 root encodings already serve outer
assignment. Sharing those encodings with sampled inner assignment is a separate
possible optimization; moving centers must always invalidate assignment
distances.

## End-to-end assignment quantization checkpoint

`arguana-quantization-packed-001` is complete and clean: four fresh independent
builds and 32 single/batch search points, forced AVX-512 through the immutable
`/tmp/mvsic-final-env` (`yaml-wheel-01`). This snapshot contains validated AVX-512
TQ4 packing and the lowest-center-ID tie fix; the later AVX2 SIMD work is not
part of this end-to-end measurement. The selected training recipe remains
`reference` (`same/default/legacy`), with eight build/batch workers, CPU0 single
search, model seed 0, and the same 128-query sample and serving policy as the
float training pilot. `review.json` preserves every curve and its limitations.

| Assignment | Build seconds | Peak RSS GiB | Recall at probes 4 / 8 / 16 / 32 |
| --- | --- | --- | --- |
| Float | 37.831 | 1.165 | .74453 / .84063 / .90234 / .94688 |
| TQ8 | 36.560 | 1.445 | .75625 / .84219 / .90391 / .94375 |
| TQ4 | 38.664 | 1.368 | .74141 / .84844 / .90313 / .93828 |
| TQ1 | 131.679 | 1.281 | .74766 / .82266 / .89141 / .93203 |

Single and batch recall agree. TQ4 packing makes this whole build comparable
to float, but its whole-build speedup over the old TQ4 path still needs that
baseline to finish. TQ1 completed successfully but is slower to build and misses
float's high-probe recall. None of these one-seed/128-query results justifies
changing the float training default.

The sequential old installed baseline, `arguana-quantization-baseline-002`,
completed float and TQ8 controls before a brief system apt index update tripped
the strict guard during TQ4 construction. Float built in 38.215 seconds with
1.092 GiB peak RSS; TQ8 built in 36.759 seconds with 1.440 GiB peak RSS. Its TQ8
recall was .75313 / .84063 / .89375 / .93047, matching the earlier old snapshot.
The changed TQ8 tree/recall must be reported separately from packing speed:
the candidate fixes lowest-center-ID ties. Peak RSS varies across these single
build repetitions; do not attribute the float RSS difference to a demonstrated
algorithmic memory change.

`baseline-002/partial_review.json` scopes only completed float/TQ8 controls as
usable. All their searches ended by 12:48:51.633 UTC, before the final quiet host
sample at 12:48:52.409 UTC. `/usr/lib/apt/methods/store` then consumed about one
core during two samples at 12:48:54/56 UTC. The affected TQ4 attempt was stopped
after about five seconds and is excluded; TQ1 had not started. The full baseline
phase remains incomplete. No external process was changed.

The completed retry used the unchanged `/tmp/mvsic-refactor-env` with
`--recipe TQ4 --recipe TQ1 --quiet-host` and fresh output
`arguana-quantization-baseline-003`, after root's combined build/test window.
The new optional filter records selected recipes and omitted parents, preserves
new artifact paths/order, and never invents comparisons to an omitted float
parent. Six planning tests pass without native work. Both fresh builds and all
16 search points completed with 118 quiet host samples and no busy sample.

The combined comparison is saved as
`arguana-quantization-packed-001/paired_review.json`. It checks identical inputs
and query selections, unchanged baseline runtime fingerprints across the two
baseline phases, independent builds, and matching serving parameters. It uses
only the completed float/TQ8 controls from `baseline-002` and the complete
TQ4/TQ1 runs from `baseline-003`.

| Assignment | Old build s | Packed build s | Speed ratio | Old / packed peak GiB | Identical artifact |
| --- | --- | --- | --- | --- | --- |
| Float | 38.215 | 37.831 | 1.010x | 1.092 / 1.165 | Yes |
| TQ8 | 36.759 | 36.560 | 1.005x | 1.440 / 1.445 | No: tie handling changed |
| TQ4 | 82.989 | 38.664 | **2.146x** | 1.367 / 1.368 | **Yes** |
| TQ1 | 131.505 | 131.679 | 0.999x | 1.325 / 1.281 | Yes |

The TQ4 result isolates an actual whole-build improvement: its serialized index
SHA256 is identical and all single/batch recalls match exactly. Its 2.146x
whole-build ratio is separate from the roughly 16x AVX-512 inner-assignment
microbenchmark ratio. TQ8's old/new recall at probes 4/8/16/32 changed from
.75313/.84063/.89375/.93047 to .75625/.84219/.90391/.94375 with the lowest-ID
tie correction; do not present it as a packing-only comparison. Float and TQ1
also retain identical serialized trees and recall; their small timing/RSS
differences are single-run observations. The float training default remains.

No native jobs remain in this quantization window. TQ1's roughly 132-second
construction motivated the separately measured assignment candidate below.

## TQ1 assignment SIMD candidate

`tq1-assignment-001` contains a clean, full-host-quiet eight-case comparison on
one physical CPU (CPU 0), with seven repetitions and alternating variant order.
The temporary source compiled against the extracted `final-build-02` native
source snapshot, never live quantizer headers. Compiler commands, binary/DSO
and dependency-header hashes, raw samples and the common host guard are saved
with the results. Both AVX2 and AVX-512 passed original scalar and independently
decoded-codec oracles before timing. Labels matched exactly; score comparisons
used `abs(error) <= 2e-6 * max(1, abs(reference))`. The prototype did not record
the maximum observed error, so these results are not a claim of bitwise score
equality. No VNNI, AMX or AVX-512 VPOPCNT path was measured.

Outer cases used 32 query clouds of 140 vectors against 94 center clouds of
140 vectors, at dimension 128, with three loops per repetition. Inner cases
used 2,800 singleton queries against 140 centers, also dimension 128, with ten
loops. Ratios compare the old single-vector scalar traversal with SIMD across
center vectors; total ratios include moving-center encoding and panel packing.

| Assignment | ISA | Metric | Kernel median ratio | Encode + pack + assign median ratio | Total min ratio |
| --- | --- | --- | --- | --- | --- |
| Outer | AVX2 | IP | 2.507x | 2.376x | 2.384x |
| Outer | AVX-512 | IP | 3.667x | 3.436x | 3.484x |
| Outer | AVX2 | L2 | 2.422x | 2.363x | 2.371x |
| Outer | AVX-512 | L2 | 3.852x | 3.508x | 3.515x |
| Inner | AVX2 | IP | 3.204x | 2.879x | 2.881x |
| Inner | AVX-512 | IP | 4.944x | 4.232x | 4.239x |
| Inner | AVX2 | L2 | 3.516x | 3.053x | 3.084x |
| Inner | AVX-512 | L2 | 5.377x | 4.619x | 4.624x |

The source integration stays private in `mvclustering_kernels.cc`. It reuses
existing AVX2/AVX-512 popcount helpers, packs changing centers once per
assignment, and retains per-vector norms. Every vector pair receives the same
float score epilogue before minima are taken; minimum Hamming distance alone
is invalid for general norms. The singleton inner layout is selected once
when constructing its assignment object. Immutable query encoding reuse,
float centroid updates, weights, sampling and seeds remain unchanged. Ascending
center traversal and explicit lowest-ID reduction retain deterministic ties.

New existing-suite regressions independently decode the codes and cover
IP/L2 norm-before-min counterexamples, zero vectors, dimensions 7/31/128/257,
ragged clouds and center counts around 8/16/32, plus cross-panel singleton ties.
The combined Clang/GCC native checks subsequently passed all five suites on
both ISAs, followed by final installed-package checks and 84 strict parity
checks. These validate the production integration separately from the earlier
temporary prototype. No VNNI/AMX validation is implied.

## Final TQ1 whole-index comparison

`arguana-quantization-tq1-simd-001` uses the final installed `final-wheel-02`
snapshot, forced AVX-512, with the same reference recipe, model seed 0, eight
build workers, 128-query sample seed 42, probes 4/8/16/32, TQ8 reranking, and
single/8-worker batch schedules as the earlier quantization runs. It builds
only TQ1 from scratch. All eight search points completed; the strict full-host
guard recorded 23 samples and no busy sample. No further native work was
launched after this window.

| TQ1 snapshot | Build seconds | Peak RSS GiB | Speedup versus this snapshot |
| --- | --- | --- | --- |
| Original assignment (`baseline-003`) | 131.505 | 1.325 | **4.241x** |
| Packed TQ4/TQ8 only (`packed-001`) | 131.679 | 1.281 | **4.247x** |
| Final TQ1 SIMD | **31.005** | 1.275 | 1.000x |

All three serialized indexes have exactly the same SHA256:
`ab16dbcf95db10f788fa55536ed610f595cdefa927d376bb63990ea967e7fcc0`.
Single and batch recall@10 also match exactly at every probe count:
`.74765625 / .82265625 / .89140625 / .93203125`.
`paired_review.json` checks identical training parameters, spec, input hashes,
query selection, serving parameters, fresh builds and quiet host records.

This is a measured 4.24x whole-build improvement, separate from the assignment
microbenchmark ratios. It is still one model seed and one build per snapshot;
small RSS differences are not a general memory improvement claim. The final
snapshot also contains serving tie/bucket fixes, so any serving-QPS difference
is not assignment-only evidence. TQ1 now builds faster than the earlier float
reference (37.831 seconds), but float retains higher measured high-probe recall
(.946875 versus .93203125 at probe 32). The float training default remains.

## Provisional serving byte accumulation and overflow correction

The subsequent changes are in serving helpers:
`core/quantization/turboquant_1bit_mv.h` and
`core/kernels/panels_avx2.h`. For 2..31 Hamming tiles, byte popcounts accumulate
before one conversion to int32 lanes. Dispatch occurs before the scoring
loops. One tile and more than 31 tiles retain the previous path; the native
AVX-512 VPOPCNTDQ path is also retained. The byte bound is `31 * 8 = 248`.
`avx2-hamming-001` exposed a one-tile regression, and the gated follow-up in
`avx2-hamming-002` checked retaining the old one-tile route.

`tq1-byte-accum-001` records the frozen AVX-512 prototype, controller/helper
copies, dependency/compiler hashes, exact integer/score oracles and seven-repeat
timing distributions on one physical CPU under the full quiet-host guard.
At dimension 128, median cached single/grouped ratios were 1.121x/1.244x;
including query encoding and broadcast setup they were 1.125x/1.157x.
Dimension 1024 uses the unchanged larger-code fallback and stayed near 1.0x.
These are bounded serving-kernel measurements, not whole-query gains. No
VNNI/AMX/VPOPCNTDQ hardware validation is implied.

The same audit found two genuine signed-overflow risks: single and grouped
AVX-512 scoring accumulated Hamming minima across all query vectors in int32.
Public inputs permit up to `UINT32_MAX` vectors with padded dimensions up to
65536, so these two totals now use int64; per-lane counts and 4/8-query reducers
remain int32. The real regression invokes both scoring functions with 32,769
antipodal queries and stride-zero buffer reuse, requiring exact mean IP score
+1. The old single path returned approximately -2.99988. Two candidate paths
and one old-reference call took 0.382 seconds in the temporary check; no giant
float/query allocation was needed. This is a correctness fix independent of
the byte-accumulation performance decision.

The general-norm TQ1 training kernels do not use the unit-norm serving minima
or its across-query totals. Their assignment, float updates, weighting and
seeding are unchanged. The AVX2 `popcount32_avx2` helper they share was factored
into byte-count and fold helpers with equivalent integer arithmetic; training
does not select the new serving accumulation path. The accepted 31.005-second
whole-build result therefore remains attributed to the earlier training SIMD
change, measured with `final-wheel-02`.

Production source is frozen at the `final-build-04` snapshot. The old AVX-512
header is retained in `final-build-03/native-source.zip` with SHA256
`2f3890c0a1c674f9ca111d6509be3f40748a4442bbd4bdbd1b8e149acc4090bd`;
an additional source/test backup and manifest live under
`/tmp/mvsic-tq1-byte-accum/preintegration`. Six-suite Clang/GCC correctness,
installed package/C++ checks and strict parity have passed, but adoption of
byte accumulation still awaits the whole-query causal comparison. Training
defaults remain unchanged, and no additional clustering experiment was run for
this checkpoint.
