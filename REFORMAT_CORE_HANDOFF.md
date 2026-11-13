# Core and compiled API handoff

## Current work

Subsequent root-final quantization, TQ8 shortlist and exact Float batching
screens fail latency or build-assignment cost gates. None changes production.
The next phase prepares the original raw SVHIVF oracle and concrete port;
the staged MVIVF source and installed runtime below remain unchanged.

The six-file change is validated and staged: exact row views, inner TQ1 input
cache reuse, and Parlay request grouping in the existing search file. Source
78d7c073 retains the opaque public API and restores flat parallel requests,
sorting and packed group boundaries with the same query order and scatter
offsets. Final GCC and installed-package checks pass. Installation:
`/tmp/mvsic-parlay-requests-env`, wheel SHA
`5a1eff7d85111b44920dd88f4d1fe15c1a2aa695a957b3f72ffa4690d069d66e`.

Source 78d7c073 passes 24 Clang/GCC native suite invocations across both host
ISAs, 12 prediction cases, installed 48 tests plus 36 subtests per ISA and both
external consumers. It changes only the facade search object on top of the
validated 65ed65ad foundation, whose combined FiQA trees retain exact Float
e174fc03 and TQ1 dad38138 bytes. The wheel replaces only the facade and RECORD
from a158c659; executable sections, relocations and layout match the performance
variant apart from runtime paths.
The staged-source audit matches all 348 compiled source files plus three
packaging READMEs exactly and excludes the unaccepted Flat integration.
AVX2/AVX-512 hardware tests pass; VNNI is compile/ELF-only
and AMX is unimplemented. The prior foundation is installed at
`/tmp/mvsic-selected-view-cache-env`, wheel a158c659.

The latest balanced ArguAna comparison passes the criteria declared before
timing. Original→CSR serial mean/p95/p99 are 2.932927→2.817973,
3.631704→3.532138 and 3.897579→3.797675 ms (−3.92%/−2.74%/−2.56%). Median-repeat
batch QPS is 15,004.961→15,010.251: essentially parity (+0.035%), and 5.09% above
source 65ed65ad's 14,283.200. Mean-repeat batch QPS remains lower than original:
14,830.733 versus 15,112.118 (−1.86%). CSR serial is also 1.93%/1.38%/1.55%
slower than the 65ed65ad control. These are bounded observations, not a universal
speedup. Recall .943954 exceeds .940967; all current prediction bytes retain
1fff7a92. Evidence: `selected-view-cache-parlay-grouping-serving192-01`.

Keep the earlier adverse source-65ed blocks: −3.35% batch versus original and
−1.63% in runtime isolation. The Parlay grouping change is separately validated;
it does not erase those observations or establish a cause for the earlier
code-placement differences. Correctness evidence: `parlay-requests-01`.

The panel-alignment screen is closed below its 10% microbenchmark threshold.
Privately aligning functions to 64-byte boundaries worsens serial latency by
5–7% and is rejected.
Neither changed production code. The preceding staged checkpoint was 89ee99ff.

## Earlier checkpoint evidence

The preceding staged private-kernel optimization uses constant four-byte TQ1 query
loads under the established64-byte zero-padding invariant. FiQA TQ1 build falls
35.494→24.352 s with exact tree bytes. Both compiler/host-ISA native matrices,
installed API/YAML/consumer checks and12 prediction-byte cases pass in
`tq1-queryword-validation-01`. New environment `/tmp/mvsic-queryword-env`;
selected staged inventory89ee99ff. VNNI compile/ELF-only; AMX unimplemented.
Float defaults and quantized-training quality are unchanged. See the main
handoff/checkpoint report for incremental build/wheel provenance.

Earlier permanent-runtime192-worker checks pass: shared-tree mean latency
1.834→1.816ms and median batch QPS19,088→19,579, with exact current prediction
bytes. Fresh own-model p32 serial mean/p95/p99 also pass at higher current recall;
batch median QPS improves1.96%. Fresh ArguAna build is6.890→6.387s with15.37% lower peak
RSS. These are bounded shared-host acceptance blocks, not universal guarantees.
See the main handoff/checkpoint report for exact scope and old-original defects.

Confirmed runtime defect (`waiter-runtime-build-01`): ISA isolation with
`-Bsymbolic` also isolates libstdc++'s atomic-wait fallback table, although the
scheduler TLS is shared. An idle four-worker pool then receives cross-library
notifications using another table. The same compiled object runs32 jobs on
one worker before a targeted diagnostic relink and all four afterward, with
all outputs preserved. All three actual backend relinks pass static binding
checks; VNNI has no hardware execution. Production now centralizes scheduler
wait/notify in existing `core/kernels/runtime.cc`, preserving private numeric
symbols without depending on libstdc++ mangling. The new regression passes both
compilers and fails against the old runtime. `wrapper-timer-validation-01` passes
seven native suites on AVX2/AVX-512 under Clang/GCC, twelve saved-tree byte cases,
installed 63 tests plus 42 subtests per ISA, original YAML and the external C++
consumer. New environment `/tmp/mvsic-runtime-checkpoint-env`; compiled source
04b9a413 plus a README-only packaging supplement, with all code/native payloads
unchanged. Installed performance remains separate. Earlier parallel timing
comparisons may contain this defect.

Latest 192-worker diagnostics and the borrowed-row experiment are in the main
handoff. The latter uses a Parlay container of const spans, preserving scan/fill,
sampling and backend float storage while avoiding unselected row ownership.
It passes exact Float/TQ1 tree hashes and all18 clustering tests, but is not
promoted because its one TQ1 wall-time pair is adverse despite lower CPU/memory.
Its preload also shares the waiter table; those wall/CPU measurements cannot
isolate copy elimination. No public API or production numeric code changed.

Fresh default build confirmation is complete in `mvivf-default-build-pair16-01`:
22.058 versus 22.472 s at 16 workers, with GNU-time peak RSS 1,980.73 versus
1,431.19 MiB. Common-runtime full-query recall is lower by 22/7 correct neighbors
at 16/32 probes, so this is not an unconditional performance pass. Refactor tree
bytes match both earlier builds. The fresh original has nonunit saved centers;
one exact short-document padding failure is proven in root center 59. Details
and source scope are in the main handoff and checkpoint report.

Latest applied worktree validation is `flat-integration-validation-01`, source
`d52f30771a654738fcf1e4ad101203571a91ab65dd4b8a9538065a899a0e0c8f`.
Seven native suites pass under Clang/GCC on AVX2/AVX-512; twelve MVIVF saved-tree
byte pairs match. The installed wheel passes 63 tests and 42 subtests per ISA,
the external CMake consumer passes, all 160 public-ID cases pass and all 40
implicit byte controls match on each host ISA. The installation is `/tmp/mvsic-flat-env`;
previous environments are preserved. Flat is applied and correctness-validated
in the worktree, but its changes remain unstaged pending performance acceptance.
VNNI is compile-only; AMX is unimplemented.

The actual installed default-float check (`flat-installed-checkpoint-01`) matches
IDs/score bits and recall, but its mean, p95 and p99 are 0.64%, 0.83% and 0.89%
slower than the original. It stops at that strict gate; TQ1 and full-query quality were not
run. Do not describe the installed Flat port as performance-accepted. MPool is
preparation-only.

The f1-to-d52 MVIVF component block (`mvivf-two-method-component-performance-01`)
preserves every prediction byte, with serial mean 2.573 to 2.308 ms and batch
5,982 to 5,886 QPS (-1.61%). Peak-memory deltas are below 1 MiB. This compares
two refactor runtimes, not the original baseline, and does not clear the earlier
shared-host or full-machine performance limits. The previous f1-only correctness
checkpoint remains archived in `public-id-final-validation-01`.

The corrected worker policy defaults ancillary library threads to1 while Parlay
uses configured workers; serial latency forces all1, with explicit provenance.

Earlier installed correctness checkpoint: `public-id-validation-01`, source
`a37e5d5080286544af3836801638906359d7d03e251398a1183d505550dd09d3`.
All six suites pass under Clang/GCC on AVX2/AVX-512, twelve saved-tree default
prediction-byte pairs match, and each installed ISA passes45 tests plus29
subtests. The external installed consumer passes both. The new isolated
environment is `/tmp/mvsic-public-id-env`; older environments remain preserved.
Custom-ID labels now survive pruning/rerank correctly. Default implicit-ID
controls retain identical ID/score bytes; separate performance acceptance
remains open. VNNI remains compile-only and AMX is unimplemented.

`SearchResultWithStats` field2 is estimated database bytes, as in all original
methods. Logical comparison counts belong to method-specific statistics;
do not silently reinterpret the installed YAML's historical `avg_cmps` column.

`public-client-compile-cost-01` completes the frozen original/installed-refactor
client compile comparison: Clang15/O3/AVX-512, one compiler,103.94→.87 s wall,
1,181,900→138,224 KiB compiler peak RSS and8,012,008→11,968-byte objects.
Both clients construct/configure and use build/search/save/load. This measures
client header parsing/codegen, excluding link and shared-library build costs;
it is not a119x whole-library build claim. Source/dependency/symbol checks pass.

November 12 API update: the public `Metric` enum and `Index::metric()` are retired.
`IndexMVIVF` now takes `(uint32_t d, const IndexParams& params)`; MVClustering
also has a concrete IP API and no `inner_metric` field. Shared private L2
k-means/codebook math remains. The original YAML accepts `metric: ip`, while
L2 requests fail at Python/config boundaries. `ip-api-validation-01` passes all
six native suites under Clang/GCC on AVX2/AVX-512, twelve saved-tree prediction
byte comparisons, both installed API/YAML suites and an external CMake consumer.
The new isolated installation is `/tmp/mvsic-ip-api-env`; both research envs
remain unchanged. Historical signatures below describe earlier snapshots.

Prepared owning query and probe containers now use correctly constructed Parlay
sequences for parallel destruction. Large-batch boundary validation transports
worker errors deterministically to the caller; no valid task takes a mutex.
The private scheduler patch bypasses adaptive grain clocks and job publication
for one-worker loops/forks. This is not a compile-time sequential build.
The modern original latency path also used a one-worker scheduler; legacy
`bench_seq` and `--config=serial` selected `PARLAY_SEQUENTIAL`.

The user confirmed keeping runtime one-worker latency after the historical
compile-sequential change was verified. The recovery queue completed on 16
disjoint physical cores in `refactor/parlay-recovery-queue-01`: all six native
suites pass on both host ISAs under Clang, and the 12-process comparison preserves
current/candidate prediction bytes. Shared-host timing scope limits speed claims.
GCC and installed-wheel refresh remain pending; the installed wheel still
contains the preceding final-build-04 implementation. The remaining work-count
diagnostic is queued at `refactor/work-trace16-queue-01`, requiring exactly 16
disjoint physical cores and guarding their SMT siblings.
See the main handoff for queue scope and constraints.

# Provisional AVX2 Hamming optimization (November 11, 2026)

Current native source is frozen for the root agent's combined rebuild. The AVX2
TQ1 helper provisionally accumulates byte popcounts for 2..31 tiles and folds them
once; one tile and larger dimensions retain per-tile int32 folding. The byte
bound is 31*8=248, and float evaluation order is unchanged. Shared panel tests
now check both packed and broadcast queries against an exact scalar oracle,
including tile counts 1/2/4/9/31/32/64 and byte/point/query tails. This extends the
same test that covers the AVX512 candidate and long-cloud sum overflow fix.

Both temporary AVX2 experiments passed all 5,880 exact checks under the shared
quiet-host guard. `refactor/avx2-hamming-001` exposed a one-tile regression; the
2..31 gate in `avx2-hamming-002` removed it (IP d31 geomean 0.997x). The gated
whole-helper d128 IP ratio is 1.100x across the measured shapes, with per-case
ratios 1.075–1.122x. Compiler factorization/layout effects also affected fallback
controls, so these results are not a pure instruction-count attribution. Final
adoption still requires the combined correctness checks and whole-query ABBA.
Raw distributions, source/binary hashes and the old production header backup are
retained in those artifact directories. No VNNI/AMX hardware claim is made.

Packed-key selection was tested separately in `refactor/selection-001` and was
**not adopted**. Its retained 453/634-candidate prerank fixtures save only median
0.74/1.10 us including fresh scratch, with material run-order sensitivity and
additional scratch/zero-representation complexity. The existing comparator stays.

# Installed serving micro checkpoint (November 11, 2026)

`/ssd2/kishen/mvsic_new/refactor/serving-micro-002` completed all 24 workers in 151.3 s
under the existing full-host HostGuard, with no busy samples. It pins the final
installed package and loaded libraries, records the final-build-03 source manifest,
and retains inputs, every prediction array, first-use costs and raw repetitions.
All four leaf codecs passed short/medium/long/ragged single-versus-batch checks;
fixed-tree TQ4-center/TQ1-leaf/TQ8-rerank root-M2M on/off passed at 32/128 queries
and 1/4 threads. All 256 within-worker checks and 18 cross-ISA comparisons passed.
Within-worker scores were bit-identical; largest cross-ISA error was 7.45e-8 and
neighbor IDs agreed throughout. VNNI/AMX were not hardware-tested.

This compares modes of the final implementation, not old versus new. Root-M2M
throughput ratios are 1.014–1.042 on AVX512 and 0.995–1.009 on AVX2. TQ1 leaf batch
versus single ratios are 1.106–1.160 on AVX2; on AVX512 short-tail is 1.094, ragged
0.999, long 0.980 and medium 0.958. The 2–4% slower AVX512 one-leaf medium/long batch
cases remain a bounded performance observation, not an attributed regression
against the original. TQ4/TQ8 batch ratios are 1.008–1.203. Original-versus-new
matched-recall acceptance still needs the controlled comparison suite.

# Bounded bucket tie fix checkpoint (November 11, 2026)

The first guarded serving micro stopped on TQ1 single/batch membership differences.
The saved reproduction in `refactor/leaf-tie-repro-001` proves all per-ID full-k
scores match bit-for-bit on AVX2 and AVX512; the differences are exact cutoff ties.
The private bounded bucket now retains the smallest `(distance, unsigned ID)`
pairs consistently during pruning, eviction and output. The equivalent TQ1
large-k, TQ4 and TQ8 heap cutoffs also compare both fields. Scoring and grouped
leaf traversal are unchanged. Native regressions exercise high unsigned IDs,
SIMD tails, equal-score arrivals, k513 heap fallback, all serving codecs and
public single/batch results.

Standalone guarded measurements in `refactor/bucket-001` and `bucket-002` compare
exact historical/current structs with correct scalar and two-pass controls.
All correct oracles passed; accepted/displacement counts match in all 45 timing
fixtures. Six raw repetitions, source/compiler/binary hashes and the shared
full-host guard are retained. Scalar reduction wins at k=10/25; above 32 the source
uses fused AVX2 distance/ID reduction and simpler AVX512 distance reduction plus
tied-ID scan. Masked tails never read inactive scratch. Explicit float/ID sorting
reduced correct helper cost about 28–30% versus default C++20 pair ordering.

Compiled MVIVF now requests an unsorted TQ1 bucket drain because `Impl::finish`
already selects and orders globally after collecting all leaf candidates.
Existing helper callers retain sorted output by default. This removes redundant
per-leaf sorting without changing the public contract; the new test also checks
that both drain modes write all sentinel slots. The combined final-build-03 passed
all five native suites under Clang and GCC with both host ISAs, eleven installed
Python tests, an external C++ consumer and 84 strict parity comparisons. The
installed final-wheel-02 also passed the complete serving micro below. Standalone
timings are not an accepted original-versus-new serving speed claim; the failed
first serving run remains incomplete and is superseded by serving-micro-002.

# Serving validation and coarse-tie checkpoint (November 11, 2026)

The latest source adds guarded eight-accumulator double norm validation at
`d >= 128`, with the original scalar path below that threshold. Near the unit
or finite-scoring norm bounds it recomputes in the original order, preserving
acceptance and error messages. A new native GoogleTest carries all 11,460
exceptional/boundary fixtures from the measured prototype. The validator passed
the coordinated final build, five native suites on both host ISAs, installed
Python tests, external C++ consumer and strict IP parity. The newer bucket tie
fix described below is still awaiting the next combined build.

Prototype results are in `/ssd2/kishen/mvsic_new/refactor/validation-001`, including
source/header/compiler hashes, executable, raw repeats, disassembly and the
full-host guard. On CPU80 with facade AVX2 flags, a 32x128 unit query's validation
median fell from 3.686 us to 0.848 us (2.839 us saved). This is about 0.158% of a
1.8 ms query and cannot explain the entire approximately 0.64% same-tree engine
gap. Unguarded four/eight sums changed 11/12 boundary decisions; both guarded
variants matched all fixtures. Small d=7 regressed, motivating the coarse
dimension threshold. These are standalone validation timings, not an accepted
end-to-end serving speedup.

Both previously unexplained cross-tree final-neighbor differences are classified
in `refactor/coarse-cutoff-01`. Original00/query109 has 453 identical visited IDs
and bit-identical coarse scores: one remaining rerank50 slot falls in the tie
group [8383,715,435], where original selects 8383 and new selects 435.
Refactor01/query11 has 634 identical visited IDs and scores: four slots in an
eight-way tie select original [6187,5492,4646,2811] versus new
[1004,2015,2116,2134]. Original `packFloatAndInt` numerical sorting makes larger
IDs sort first at negative equal scores; new ordering uses ascending IDs. The
changed coarse cutoff membership explains distinct, non-tied final TQ8 results.
There is no traversal/coarse-score discrepancy for these rows and no padding-bug
exception. Full candidate outputs and classification scripts are retained.

The packed inner-assignment AVX2 candidate in `turboquant_8bit_mv.h` now uses the
shared byte-panel dot helper, two queries per block, eight-center half-panels,
explicit real-center masks and lowest-ID tie reduction. Root-verified expanded
micro-oracles passed on AVX2/AVX-512. Clean `inner-assignment-005` measurements
reported approximately 11.4-11.7x gains for the previously scalar AVX2 packed
assignment cases; the clustering agent controls production TQ4 packing and its
end-to-end training verification. VNNI/AMX hardware validation is not claimed.

The installed-API serving experiment's first attempt, `serving-micro-001`,
stopped on a TQ1 single/grouped top-k mismatch; its timings are not accepted.
`leaf-tie-repro-001` then proved bit-identical full scores on AVX2/AVX-512, with
all 12 affected query rows differing only in exact cutoff-tie membership. The
TQ1 bounded bucket evicted the first maximum-score slot instead of the largest
ID tied at that score. The latest uncompiled fix makes pruning/eviction/finalizing
use (distance,rowID), with one SIMD maximum pass and safe masked tail loads.
TQ1's large-k fallback and TQ4/TQ8 heaps also accept smaller-ID tied arrivals.
Native regressions cover partial buckets, high unsigned IDs, descending tied
IDs at k7/k513, and public single/grouped search with k below leaf size. These
source changes require the coordinated incremental build before the microbenchmark
resumes. Its driver now uses the shared full-host HostGuard and persists every
prediction record before assertions. It does not substitute for the controlled
original-versus-new comparison.

## Previous source checkpoint

# IP-only retrieval cleanup (November 11, 2026)

The user retired L2 retrieval while retaining L2 clustering and shared distance
math. `IndexMVIVF` now rejects non-IP metrics with an IP/MaxSim explanation; its
ISA factories instantiate only the IP serving specialization. Private serving
factory signatures no longer carry a redundant metric argument. The native
index test checks L2 rejection, while independent L2 codec/panel tests remain.
Serving `Query`, `TypedView`, and `IPKernels` are now concrete IP types; there
is no private metric field or L2 serving branch. The public constructor retains
the metric argument to reject unsupported requests, and tree training explicitly
uses IP outer assignment while keeping the independent inner metric setting.
The root agent coordinates fresh compilation and installed-package validation;
the November 10 results below describe the preceding source snapshot.

Removed the superseded regular legacy index header and its remaining coupled
diagnostics: the regular `microbenchmark/mvivf` batch instantiations, leaf
diversity and old/new chamfer-scoring tools, the clone's MVIVF bucket pilot, and
the tree-quality wrapper. Their historical sources remain in the original
checkout. Independent kernel/sort/dedup tools and the Vamana bucket pilot remain.
The stage-summary plotting script no longer calls the retired tree-quality tool.

Flat/Spill IP algorithms remain. Their 80 L2 generated benchmark translation
units (including Spill microbenchmarks) and L2 aliases were removed; regenerated
IP translation units are byte-identical. The shared retrieval CLI macro and four
remaining direct legacy CLI dispatchers now reject L2. Other method L2 public
aliases were retired without changing their IP implementations or internal L2
transform/training code. The active clustering trainer was renamed separately
to `mvclustering_impl.h`; it is not an obsolete implementation to delete.

Source checks: 37 BUILD files and changed Python generators parse; no missing
current local includes; `git diff --check` passes. Optional Flat/Spill and other
legacy binaries have not been rebuilt. Low-level quantization diagnostics still
exercise L2 scoring, and the clustering benchmark retains both metrics.

## Previous compiled checkpoint (November 10, 2026)

The compiled runtime API is implemented and native correctness is verified.
The earlier unbuilt draft below is historical context, not the current state.
Performance acceptance is still pending; do not claim parity or speedup from
correctness tests. The root agent coordinates builds and timing windows.

- Public API: `mvsic/algorithms/mvivf/mvivf.h`, runtime `Metric`, owned or explicitly
  borrowed `PointCloudSetView`, compiled build/search/persistence and coarse ISA
  dispatch. Quantized centers and None/TQ1/TQ4/TQ8 leaves, grouped batch traversal,
  root M2M, and lazy TQ8 reranking are implemented. V4 skeletons load from the
  original checkout; IDs are translated through dense internal rows.
- Boundaries reject malformed shapes/offsets, duplicate or reserved UINT32_MAX
  IDs, nonfinite/overflow-prone values, invalid configs, and nonunit serving TQ1
  input. Root corpus ownership survives caller destruction; rebuild/load replaces
  kernels and lazy caches transactionally. Clustering assignment has independent
  norm-aware codecs and float updates.
- Shared Parlay scheduler TLS is exported once by `core/kernels/runtime.cc`.
  The private dependency overlay uses global-dynamic TLS for PIE executables and
  Python dlopen. AVX2/AVX-512/VNNI serving and clustering DSOs share the current
  scheduler. Scheduler tests cover one and three workers across two DSOs.
- Correctness: Clang native suites pass with AVX2 and AVX-512 selected, including
  raw exact IP/L2 oracles, sparse IDs, ragged batches, codec/rerank consistency,
  compression, ownership, persistence, and invalid inputs. Logs are under
  `/ssd2/kishen/mvsic_new/refactor/clang-tq4-validation`. The installed wheel
  API/CLI checks pass under `refactor/wheel-validation-02`, and an external C++
  CMake consumer passes under `refactor/cpp-install-01`.
- Fixed an inherited **TQ4** AVX-512 IP padding bug: padded zero-scale lanes could
  beat real positive negative-dot distances. Negative padding scales and a mask
  using the already loaded scales preserve real zero vectors. Older encoded
  blobs upgrade their padding metadata after offset validation. TQ8 already had
  correct padding handling and its numerical code was not changed for this fix.
  `core/kernels/panels_test.cc` independently decodes actual panel points and
  tests antipodal/zero inputs, full/tail panels, and grouped query blocks through
  both serving backends.
- Strict parity: `refactor/parity-03` passes all 168 checks against untouched
  original AVX2 single-query output, with maximum score error 4.77e-7. All new
  AVX2/AVX-512, one/three-worker, single/batch/root-M2M modes agree, including
  partial visited sets and compression. Original AVX2 TQ1 batch has a placeholder
  fallback, so the explicit `--reference-single-only` option excludes that
  baseline self-check only. Native AVX-512 remains the performance baseline.
- Safe memory/encoding changes: primitive scratch is uninitialized only where
  every consumed element has one complete writer; owning objects and accumulators
  retain initialization. Serving quantized leaves encode row-mapped views without
  float gathers. TQ4 center/leaf queries share one encoding; serving TQ8 queries
  are reused for reranking only when the effective input matches. Initial TQ8
  database preparation remains lazy and must be accounted for separately.

Cleanup completed in source after those correctness checks: removed 72 generated
regular-MVIVF benchmark TUs, their generator/extern/dispatch entries, and obsolete
regular-only `bench_seq.cpp`/`build_index.cpp` wrappers. Optional mixed legacy
benchmarks now require Flat or Spill and point regular users to installed
`mvsic benchmark` commands. Flat/Spill generated files remain. Regeneration and
local include/reference checks pass; optional legacy binaries have not been
rebuilt during the active performance window.

The legacy regular header is intentionally retained for useful diagnostics that
still access private tree state: `microbenchmark/mvivf` leaf-scoring/diversity and
legacy batch experiments, plus the clone's EMDIVF bucket pilot. Porting them would
require a separate diagnostics interface; do not widen the public Index API just
to delete that header. The original repository is unchanged. The benchmark agent
owns controlled comparisons; no independent heavy work while its timing runs.

## Historical unbuilt draft

# Compiled MVIVF core handoff (WIP, unbuilt)

Checkpoint: work stopped at the parent's request before any new-clone compilation. No commit, push, or staging was performed by the core agent. The code below is deliberately incomplete and is not a working native library yet. Continue in `/home/kishen/MVSIC-new/MVSIC`, branch `reformat`; the original experiment repo remains `/home/kishen/MVSIC`.

## User direction and agreed API

Speed first; keep existing class/function names and comment style. Simple public modules under `mvsic/{core,algorithms,benchmark,python}`. The new public `Index` and `IndexMVIVF` are non-template, runtime `Metric`; private metric-specialized numerical kernels are allowed. Do not merely wrap all 36 legacy metric/codec/center index instantiations. Preserve the production TQ4 centers + OneBitTQ leaves + TQ8 rerank and batched root-M2M / leaf-M2M paths. Tree MVIVF is the first scope; other algorithms remain optional legacy code. No performance claims have been measured for this redesign.

Public headers installed:

- `mvsic/core/types.h`: runtime `Metric {IP,L2,InnerProduct=IP,SquaredL2=L2}` and `ClusteringQuantizer {None,TQ8,TQ4,TQ1}`.
- `mvsic/core/data.h`: metric-neutral `PointCloud` aggregate `{const float* values, uint32_t n,d,id}`. `PointCloudSetView` aggregate `{uint32_t n,d; const float* values; const size_t* offsets; const uint32_t* ids; shared_ptr<const void> owner}`. Offsets count FLOAT COORDINATES, matching existing PCS files. Owner must retain values, offsets, and IDs. The index must retain the bound corpus view/owner during build/load.
- `mvsic/core/index.h`: non-template abstract base with virtual destructor, `build(view)`, `search(query,points,SearchParams)` and bound-corpus overload returning `vector<pair<uint32_t,float>>`; `search_all(queries,points,params)` and bound overload returning nested vectors; `search_with_stats` returns tuple of neighbors, bytes, timer vector; save/load(path,points), metric(), get_dims().
- `mvsic/algorithms/mvivf.h`: final pimpl `IndexMVIVF(uint32_t d,Metric,const IndexParams&=IndexParams::mvivf())`, move-only, same methods plus tree diagnostics and params(). HEADER DECLARATIONS ONLY; methods are still undefined.

`IndexParams::mvivf` signature remains unchanged (13 arguments). It now sets `pq.method=OneBitTQ`; `quantize_centers=true` remains its default. Existing SearchParams scalar fields now have initializers. Runtime leaves drafted in the private kernels: None, OneBitTQ, TurboQuant (4-bit), EightBitTQ. Other enum choices should throw clearly. Root's new Python bindings already target this public API and expose quantizer through `p.pq.method`.

## Files implemented by core agent

- `core/data.cc`: checked PCS loader (mmap and owned-copy), mapped-file RAII unmap, owned vectors, gather helper, basic ragged shape validation. Uncompiled and untested. Mapped offsets are copied because their on-disk location may not be size_t-aligned. IDs omitted imply row IDs. Input finite values and uniqueness of IDs are not yet scanned; decide checks at the right build boundary.
- `core/index.cc`: out-of-line virtual destructor.
- `algorithms/mvivf_internal.h`: metric-neutral Node (children, dense corpus rows, raw centers), encoded handles, prepared-query interface, Kernels block interface, SearchStats, proposed IndexMVIVF::Impl. All search/build member declarations remain undefined.
- `algorithms/mvivf_kernels.cc`: first uncompiled draft of exactly two private metric kernel implementations, with runtime leaf choice. It trains TQ4 centers and selected leaf model, eagerly prepares TQ8 rerank DB against bound corpus, encodes each leaf once, reuses query compression, has scalar/existing OneToMany raw leaf scoring, root fused TQ4 M2M and quantized leaf M2M, and exact/TQ8 rerank. This is not yet registered in a build target. `neighbor_less` and `select_neighbors` are declared but undefined until search.cc is implemented.
- `core/BUILD`: new compiled :data and :index and header :types rules. Original heavy :index rule became :legacy_index.

Legacy preservation was mechanical, to allow the new public names to coexist in parity tests without compiling the old codec Cartesian product into the new extension:

- Original `core/index.h` copied to `core/legacy_index.h`, class/constructors renamed `LegacyIndex`.
- Legacy family includes now point to legacy_index.h and all `Index<...>` uses became `LegacyIndex<...>`.
- Old `mvsic/algorithms/mvivf/legacy/mvivf.h` class renamed `LegacyIndexMVIVF`; legacy aliases retain names, and generated benchmark references updated to the renamed class. Existing generated source files remain optional legacy targets, not a proposed new extension source list.
- Existing family BUILD dependencies changed from core:index to core:legacy_index.
- Some legacy comments changed mechanically along with symbol replacements. No legacy numerical algorithm was intentionally modified.

Root and the experiments agent own other files currently changing in the clone. Do not overwrite their Python/CMake/benchmark/clustering work. In particular root owns `mvsic/python/bindings.cpp`, root packaging, and benchmark modules; experiments agent owns compiled clustering and `mvsic/core/kernels` ISA scaffold.

## Immediate next steps (no build has run)

1. Read all handoff files and current git diff before editing. Coordinate finished clustering API from its separate handoff. Agreed interface was `ClusteringResult cluster_point_clouds(const PointCloudSetView&,uint32_t k,uint32_t s,Metric,const MVClusteringConfig&)`, result `{PointCloudSetView centers; vector<uint32_t> assignments;}` for ALL input rows. Assignment quantizer independent of leaf quantization, raw float centroid updates. The experiments agent planned a callback in the existing private float trainer to preserve its update/seeding loop, plus runtime TQ8/TQ4/TQ1 assignment and safe general-norm TQ1 scalar kernels. It owns algorithms/BUILD :mvclustering; append :mvivf only after checking that file.
2. Implement `algorithms/mvivf_build.cc`: Impl constructor/validation/build, metric-neutral recursive topology, prepare codecs, diagnostics, checked save/load. Use one coherent dense-row mapping internally and translate to public IDs only in final results. Bind owner. Enforce same corpus storage in explicit points overloads so cached TQ8 cannot silently score a different corpus. Guard empty clouds, leaf-only root, collapsed clusters, depth limit, oversized params. Avoid repeated corpus copies; transient gathered leaf raw storage may be released after encoding except raw mode.
3. Implement `algorithms/mvivf_search.cc`: IntervalHeap beam/top probes (existing util), single search and timer vector order, sorting/selection helper, grouped batched leaf requests with fixed disjoint scatter ranges (or checked atomics), root M2M, rerank, output public IDs. Preserve old fallback for nprobes<=2 if it wins. Keep raw query for rerank unless compress_rerank=true. Sorting should be deterministic for ties. Check multiplication overflows in batch scratch sizes; return <=available rows rather than pad fake results.
4. Prefer v4 skeleton compatibility with old MVIVF for strong parity tests: magic0x4d564946,version4,class0; 8 uint32 params (k_per_level,max_leaf_size,max_depth,num_spill,s,mvclus.niters,max_point_clouds_per_cluster,max_points_per_centroid_inner_kmeans), then size_t node_count and center_offset_count, center_offsets/float centers, node_count+1 children offsets/size_t child IDs, node_count+1 point offsets/uint32 external document IDs. See legacy mvivf.h lines1550+. Validate all array sizes/offsets/file lengths/topology/dimensions/ID lookups before allocation or linking. Metric is absent in v4; bind constructor metric and document that contract or add new version with explicit metadata. Do not assume external IDs are contiguous (legacy loader did).
5. Add algorithms/BUILD compiled :mvivf sources and private deps. Expected production CMake source list: core/data.cc,core/index.cc,algorithms/mvivf_build.cc,algorithms/mvivf_search.cc,algorithms/mvivf_kernels.cc,algorithms/mvclustering.cc plus concrete CPU kernels. Third-party deps: Eigen, ParlayLib, seeded RaBitQ rotator; legacy clustering additionally currently pulls ParlayANN and external KMeans. Root handles FetchContent/pins/seed patch and scikit-build. Do not add generated legacy instantiation source globs to new library.
6. Compile and correct draft errors, then meaningful synthetic tests: (a) raw IP/L2 vs independent brute force when all leaves probed; (b) build old raw skeleton, load same persisted skeleton in legacy TQ4/TQ1/TQ8 and new index, compare ordered IDs and float tolerance on no-tie unit-normalized ragged d128 inputs; (c) single/batch/root_m2m on/off and rerank exact/TQ8; (d) query compression; (e) rebuild/load invalidates caches; (f) lifetime after dropping input owner and moved index; (g) leaf-only, duplicate collapse, invalid input/params and malformed/truncated save; (h) noncontiguous external IDs. Legacy code should be linked only into parity tests.
7. Compile-time measurements: clean/cold build, incremental public-header/code changes, peak RSS, source count and binary size against old production target. Runtime: fixed persisted skeleton, identical models/seeds/input/threads/NUMA, QPS+recall and stage times, warm rerank and cold encode separate; test portable baseline and native actual host. Never claim runtime dispatch is free at dot granularity; block dispatch only.

## Draft-specific caveats to inspect first

- None of the new C++ files has been compiled. Interface/dependency and include errors may remain. In particular private quantizer headers were previously reached through huge legacy includes; add each actual needed dependency directly instead of reintroducing legacy_index.h into the new extension.
- TQ1 has a correct scalar one-query scorer but its existing `score_one_db_cloud` non-AVX512 M2M branch returns max sentinels (legacy placeholder). Draft `leaf_topk` explicitly avoids its M2M on portable builds and uses one-query scorer. Keep this correctness guard or implement a real scalar batch path.
- Existing TQ1 multi-vector scoring assumes unit-norm vectors. Preserve semantics for production data; do not silently normalize user data or assume general-norm correctness. The separate clustering backend uses a norm-aware kernel and must not accidentally use this leaf kernel for general data.
- Draft whole-corpus TQ8 eager encoding changes where preparation cost is paid (build/load instead of first reranking query). This is intentional to avoid stale once_flag and concurrent lazy mutation, but report/tune the cost and consider an explicit preparation plan/cache policy later.
- EncodedNode::raw_leaf.ids borrows Node::rows storage, so nodes/rows must stay fixed after prepare and must outlive encoded handles. Encoded quantizers copy IDs. Move/rebuild ordering needs review.
- Native-vs-portable kernel isolation/host ISA gating is not integrated for these large existing codec headers yet. The experiments agent's safe capability scaffold should select prepared operations once; CPUID never inside scoring loops. Unsupported AMX must be explicit, never auto-selected. A global -march=native wheel is machine-specific and must remain an explicit research build option.
- Legacy compute_leaf_flat_ dereferences null parent on root collapse; new implementation must handle root leaf directly. Legacy distances_subset allocated n uninitialized comparison counters but wrote only subset length before reducing all n; new rerank avoids that helper entirely.

Use all future Bazel commands with `--output_user_root=/ssd1/kishen/cache/.cache/bazel`. Do not use mutable bazel-bin paths for benchmark provenance. No experiments should contend with the separate EMD host timing windows. The core agent stopped and will make no further new-clone edits after this handoff.
