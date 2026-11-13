# MVIVF refactor checkpoint evidence

Status: source78d7 passes full correctness/package checks and its declared serial/
median-batch serving gate. The verified six-file checkpoint is staged. Earlier
adverse throughput samples and the lower mean-batch result remain explicit.
Optimization research continues; the separate Flat port remains unaccepted.
All results below are under `/ssd2/kishen/mvsic_new/refactor`.
No commit or push is authorized or has been made.

## Parlay batch request grouping

The current-core phase probe isolates serial request construction at1.924 ms
(about2.06% of a timed batch). The original implementation grouped flat jobs
with Parlay; the refactor had serial per-leaf vector growth. The one-file
replacement uses uninitialized trivial requests, exclusive scans, a total
(node,query,offset) sort and packed group boundaries. It preserves query/group8
order, candidate offsets, disjoint scatter, root many-to-many scoring and serial
search arithmetic. No public API, new translation unit or tuning knob is added.

`selected-view-cache-parlay-grouping-serving192-01` uses original/65ed/78d7 in
balanced A,B,C,C,B,A order, serial1 first and conditional batch192, three repeats
per process. The acceptance criteria were declared before timings.

| Metric | Original | Source65ed | Parlay source78d7 |
|---|---:|---:|---:|
| Serial mean, ms |2.932927|2.764735|2.817973|
| Serial p95, ms |3.631704|3.484054|3.532138|
| Serial p99, ms |3.897579|3.739840|3.797675|
| Median-repeat batch QPS |15,004.96|14,283.20|15,010.25|
| Mean-repeat batch QPS |15,112.12|14,431.32|14,830.73|
| Serial recall@10 |.940967|.943954|.943954|

The declared gate passes: serial mean/tails beat original and median batch QPS
matches original (+.035%) while improving5.09% versus65ed. This small original-
relative margin is parity in this block, not a robust throughput speedup. Mean
batch QPS remains1.86% below original; serial mean/p95/p99 are1.93%/1.38%/1.55%
slower than65ed. Keep these tradeoffs and every raw sample. All current serial/
batch predictions retain SHA1fff7a92 and common returned-ID score bits match.
Average process peak RSS is about845MiB serial/1,316MiB batch, versus original
1,484/1,745MiB; these include loading/preparation. Both models are existing
frozen own-model trees; no training or probe grid accompanies this comparison.

Source78d7 adds this one.cc change to the five exact build changes below.
`parlay-requests-final-validation-01` completes 24 Clang/GCC native suite
invocations across both host ISAs, 12 exact saved predictions, installed
48 tests plus36 subtests per host ISA and both C++ consumers. Only the changed
facade search object is rebuilt; unchanged objects/libraries have explicit
reuse and input proofs. VNNI remains compiled/ELF-only, AMX unimplemented.
The incremental wheel SHA is
`5a1eff7d85111b44920dd88f4d1fe15c1a2aa695a957b3f72ffa4690d069d66e`;
installation `/tmp/mvsic-parlay-requests-env`. Only libmvsic.so and RECORD
change from the normal a158 wheel. All allocated sections, executable addresses
and relocations match the benchmarked private facade except the intended
RUNPATH string; other payloads remain exact. The package excludes unaccepted Flat.
The staged-index audit verifies every one of348 selected source blobs and three
packaging READMEs against compiled78d7. The six new source files and reports are
staged; the separate Flat port remains unstaged. No commit or push was performed.

## Combined exact row views and TQ1 code reuse

The applied five-file implementation removes unnecessary owning row copies
before inner sampling and reuses immutable root TQ1 query codes in inner kmeans.
Parlay scan/fill order, hash sampling, duplicate occurrences, weights, encoder
seed/norm semantics, float updates and changing-center scoring are preserved.
Two regressions in the existing clustering suite cover borrowed lifetime/order
and prepared versus fresh weighted ragged/repeated samples, including zero-data
reseeding and returned-center ownership. No new translation unit is introduced.

`selected-view-cache-validation-01` validates the exact selected checkpoint:
348 source files, SHA65ed65ad, plus three separately hashed packaging READMEs.
Normal Clang/GCC builds take138.98/131.29 s including native test targets;
these are not client-compilation or original-library speed ratios. All three
ISA backends compile. Six suites on both host ISAs under both compilers give
24 passing invocations, with21 clustering cases each. Twelve saved-tree
prediction cases, installed48 tests plus36 subtests per host ISA, the original
YAML workflows and both external consumers pass. VNNI is compile/ELF-only;
AMX is unimplemented. All prior environment payloads remain unchanged.

The normal selected-source wheel is SHA
`a158c659a9831d30a386c68f8ece7838f15ec60ca983fd0f4c578407ecf8114a`;
installation `/tmp/mvsic-selected-view-cache-env`. It excludes the unaccepted
Flat integration. Combined installed FiQA gates retain exact Float e174fc03
and TQ1 dad38138 trees, with one observed build each47.032/24.358 s. These are
exactness witnesses, not a paired combined speedup. The independent individual
screens establish Float49.101→45.761 s with23.29% lower peak and
TQ128.107→25.867 s with21.98M avoided inner query encodings. Details and the
separate private quality/latency experiment are in TRAINING_EXPLORATION.md.

Final installed original/current ArguAna serial passes: mean2.877969→2.783580 ms,
p953.570705→3.501938 ms and p993.808124→3.756074 ms; recall improves
.940967→.943954 and every current prediction retains exact1fff7a92 bytes.
Conditional192-worker batch has an adverse result: average median-repeat QPS
15,060.10→14,556.25 (−3.35%), mean-repeat QPS −2.76%. Dropping first repeats
still leaves−2.97%; do not dismiss it as the old first-batch transient. Full
raw distributions are in `selected-view-cache-serving192-01`. A same-driver,
same-model previous-current versus selected-runtime block also observes1.63%
lower throughput (−1.50% excluding first repetitions), with overlapping raw
ranges but separated process medians. All predictions remain exact. Original
and direct-runtime adverse blocks are retained; throughput is not accepted.
Read-only codegen checks find equivalent serving instructions/relocations/data,
byte-identical core DSOs and unchanged Parlay scheduler sources. Linked code
placement changes, including464/573 AVX512 serving function starts modulo64.
Private panel-storage/function-alignment screens are investigating; these five
new files remain unstaged. Do not call the adverse result noise without evidence.
The first prediction attempt stopped on a missing copied fixture before any
output prediction file existed; its failed log/status and exact-input recovery
are preserved. Passed Clang suites were not repeated or rebuilt.

## Exact TQ1 query-load optimization

The selected checkpoint now also includes a private two-line query-load change.
Model query buffers already have zero padding to a multiple of64 bytes, so each
tile can use a constant four-byte copy. The database-tail copy and all score
arithmetic remain unchanged. This removes actual `memcpy` calls from both ISA
scoring loops; it changes neither the assignment policy nor the training codec.

`tq1-queryword-override-01` compares one current-source FiQA TQ1 build pair at192
workers without profiling. Wall time falls35.493990→24.351948 s (31.39% shorter),
CPU6310.245→4194.228 s, and peak RSS changes by+0.0845%. Both saved indexes are
exactly224,221,500 bytes with identical SHA256
`dad38138ed8092d6b2867571a1ca88f93d260609fd18d8cf75c1c429ebefc86e`.
The earlier TQ1 recall deficit remains; Float is still the training default.
This is a gain within existing TQ1 behavior, not an equal-recall comparison with
Float or a sub-ten-second FiQA result.

Actual-Model release checks compare831k score bits plus labels across both host
ISAs. Separate literal-kernel ASan/UBSan and protected-page boundary tests pass
on both ISAs. The attempted full-Model sanitizer compilation reached its120 s
cap; that incomplete attempt remains preserved and is not a runtime failure or
a full-Model sanitizer pass. Whole-scoring-loop microbenchmarks improve1.73–2.31x;
these are not predictions of whole-build scaling.

`tq1-queryword-validation-01` validates source3028d477: all28 native suite
invocations across Clang/GCC and AVX2/AVX-512, twelve exact saved-tree prediction
comparisons, installed63 tests plus42 subtests per ISA and both C++ consumer
runs. Only changed clustering TUs were recompiled; unchanged objects were hashed
and reused. The installable wheel incrementally replaces three ISA libraries
and RECORD. All other payloads remain identical; ELF text/relocation proofs,
exports, dependency hashes and actual loader mappings pass. New environment:
`/tmp/mvsic-queryword-env`. VNNI is compiled/ELF-checked only; AMX is unimplemented.

Selected staged inventory89ee99ff differs from the preceding52d34ba4 checkpoint
only in the clustering kernel file. Its six-suite build retains no pending Flat
dependency. The separately validated worktree wheel includes the unstaged Flat
port; source equivalence is not a claim of whole-package binary equality.

## Latest 192-worker acceptance checkpoint

The permanent shared-core wake-up fix passes both shared-tree and fresh-own-tree
comparisons. These are one controlled block each on the shared, unpinned host;
preserve all raw samples and the small margins. Float training remains default.

| ArguAna, fresh own-model measurement | Original | Installed refactor |
|---|---:|---:|
| Build/preparation, s |6.890433|6.387412|
| Reported build peak RSS, MiB |3,569.62|3,021.05|
| Whole-process CPU, s |1,263.36|1,169.07|
| p32 serial mean, ms |2.472633|2.434102|
| p32 serial p95, ms |3.083349|3.078653|
| p32 serial p99, ms |3.325000|3.293615|
| p32 median-repeat batch QPS |16,457.04|16,779.47|
| p32 mean-repeat batch QPS |16,505.57|16,657.87|
| p32 serial recall |.940967|.943954|

`mvivf-default-build-pair192-01` contains one build each,192 Parlay workers,
ancillary limits1, AUTO sizing, identical default seed/weights/iterations and
no compilation/retry. The installed tree exactly retains SHA62487820 across the
earlier16/192-worker builds. The new original tree differs again; it contains341
zero center rows, while the current tree contains none. Full document coverage
passes. Common-current-runtime recall is lower for the new tree at p16
(.909175 versus.910526), higher at p32 (.943954 versus.940541). Do not imply
quality dominance at every probe budget or reproduce invalid original padding.

`mvivf-own-tree-p32-serving192-01` uses each fresh tree with its own implementation,
all 1,406 queries and serial1 ABBA/three repetitions. Its predeclared serial gate
passes mean/p95/p99 and recall against the maximum original serial recall; only
then does batch192 ABBA run. Current prediction bytes are identical across all
four processes/modes (SHA1fff7a92). Different learned models legitimately return
different candidates; all common returned-ID score bits nevertheless match.

`mvivf-wrapper-installed-serving192-01` isolates serving on the same original
tree: mean/p95/p99 improve0.98%/1.10%/1.39%, median/mean-repeat batch QPS improve
2.57%/2.06%, and serial/batch peak RSS falls42.82%/21.16%. All current prediction
files retain exact historical SHAee262af7. Original TQ4 padding/cutoff ties still
limit cross-engine exact-equality claims; current output is stable. The earlier
111–115ms first-timed installed-batch transient does not recur in this block.

Together these clear the tested MVIVF checkpoint at comparable/higher p32 recall;
they do not establish universal speed/quality dominance or accept the separate
Flat port. Detailed reports/raw trials remain in those three SSD directories.

These six-second build measurements are for ArguAna (8,674 documents), not
FiQA (57,638 documents). The separately profiled FiQA reference takes 45.880 s
with Float training and 35.618 s with coupled outer/inner TQ1. TQ1 loses about
3.8 recall percentage points at p32 and uses more memory, so it remains
experimental. FiQA has not reached the sub-ten-second target.

Before the query-load change, selected staged inventory SHA52d34ba4 had executable/packaging
files equal the previously validated f1 MVIVF checkpoint plus nine changes:
eight non-CMake files are byte-identical to the tested 04b9 code, and CMake adds
only the 30-second runtime-test timeout. The staged build retains six native
suites; no pending Flat dependency or deleted YAML reference leaks into it.
The 55-file quoted-include closure is complete. This source-equivalence audit
does not claim whole-package binary equality with the two-method04b9 wheel.
Evidence: SSD `staged-runtime-checkpoint-audit-01/REPORT.md`.

## Shared-runtime wake-up investigation

`waiter-runtime-build-01` confirms that `-Bsymbolic` isolates libstdc++'s
atomic-wait fallback bookkeeping across ISA libraries, despite shared Parlay
scheduler TLS. A bounded same-object experiment completes32 callbacks using
one of four workers before a narrow diagnostic relink, and all four afterward.
All three actual backend relinks preserve exports and private codec/Eigen
bindings while adding only the shared waiter-table relocation. VNNI receives
static checks only. The production fix will use shared-core wait/notify to avoid
a compiler-internal-symbol ABI.

The permanent shared-core wrappers are now applied. The exact current regression
passes under Clang/GCC and fails against the preceding runtime after idling.
`wrapper-timer-validation-01` validates compiled source 04b9a413: all seven native
suites on both host ISAs under both compilers, twelve exact prediction cases,
installed 63 tests plus 42 subtests per ISA, original YAML and external C++ consumer.
New environment `/tmp/mvsic-runtime-checkpoint-env`. A docs-only wheel repair
adds the omitted benchmark README; all existing code/native payloads remain
byte-identical, verified in the packaging supplement. Full correctness builds
at four compiler jobs take231.82 s Clang and263.48 s GCC, including dependencies,
all three ISA libraries, tests and driver (plus bindings for Clang). These are
not standalone library/client compilation times. VNNI has no native execution.

`waiter-runtime-arguana-01` passes exact Float/TQ1 tree and actual-library gates
without a preload. Float wall is6.528/6.402 s before/after; TQ1 is5.319/5.282 s.
CPU and peak memory are similar. This is one diagnostic pair per mode, not a
speed median; the long diagnostic stall does not recur in either control.
Serving validation is recorded in the latest acceptance block above.

Audit correction: a section-dump command omitted `objcopy`'s explicit output
file and rewrote section-name metadata in the temporary installed backends.
The serving preflight caught their whole-file hash mismatch before any native
run. Pure Python ELF comparison proves all loadable bytes and program headers
unchanged; the three installed files were restored from the exact pinned wheel,
with their original whole-file hashes verified. Preserve the recovery record
`waiter-runtime-build-01/objcopy-recovery.json`. The no-preload build gate used
the metadata-rewritten files and records those actual identities; do not claim
that the installed whole files were untouched throughout that experiment.

This discovery limits earlier parallel timing attribution. The borrowed-row
preload also shares the table, while the serving phase-trace preload has a local
table. Preserve their raw observations and exactness checks, but do not infer
copy-only gains or an allocation-only cause of the serving transient.

`clustering-timer-validation-01` passes all19 current tests on AVX2/AVX-512;
the focused regression fails against the old trainer as expected. Assignment
and update clocks are disjoint in the new installed checkpoint. Earlier installed
environments retain the old timer: their update diagnostics include assignment
and require subtraction; build times, tree hashes and recall remain unaffected.

## Fresh default MVIVF build confirmation

`mvivf-default-build-pair16-01` uses the frozen original binary and the actual
validated installed d52 libraries, one build per implementation, 16 unpinned
Parlay workers and ancillary pools limited to one. Raw corpus/query/truth hashes
match the historical baseline before and after execution. Runtime mappings,
saved v4 structure/coverage and all quality outputs pass their checks.

| Measurement | Original | Installed refactor |
| --- | ---: | ---: |
| Build/preparation seconds | 22.057826 | 22.472365 |
| Save seconds, excluded from build | .025322 | .023753 |
| Process user/system CPU seconds | 346.57 / 3.11 | 351.69 / 2.52 |
| GNU-time peak RSS MiB | 1,980.73 | 1,431.19 |
| Prepared statm RSS MiB | 2,020.41 | 1,358.71 |
| Tree nodes / leaves | 448 / 409 | 434 / 397 |
| Center vectors | 63,164 | 61,188 |
| Common-runtime recall, 16 probes | .910740 | .909175 |
| Common-runtime recall, 32 probes | .944452 | .943954 |

Build includes boundary checks, training, TQ4-center/TQ1-leaf preparation and
temporary destruction; it excludes PCS loading, construction, save and lazy
TQ8 rerank encoding. Both fresh trees are served by the same installed runtime
for quality only, avoiding the original AVX-512 TQ4 padding defect as a scoring
confound. No serving timing is collected. The refactor's build point estimate
is 1.88% higher, peak RSS is 27.74% lower, and recall loses 22/7 correct neighbors
out of 14,060 at the two fixed probes. One shared-host pair is not a speed median
or unconditional nonregression pass. Raw statm and peak counters disagree in
the original process; keep both as reported process-memory measurements.

The installed tree is byte-identical to both earlier leaf100 refactor builds.
The original tree differs from its earlier builds. Offline inspection proves
an original padding defect in root center 59: both trees represent document 7264
(19 vectors), but original stores 19 exact source vectors plus 121 zeros;
refactor repeats those source vectors modulo 19 throughout all 140 slots.
All 468 original nonunit center vectors and the refactor's unit-center audit
are recorded separately. This is actual content change, not EPS or a label
permutation. It does not attribute the whole recall difference to that one
defect; original global reseed counters remain a possible additional contributor
without causal attribution in this run.

## Latest applied Flat/MVIVF correctness and installed performance checkpoint

`flat-integration-validation-01` archives source
`d52f30771a654738fcf1e4ad101203571a91ab65dd4b8a9538065a899a0e0c8f`.
All seven native suites pass under both Clang and GCC on AVX2 and AVX-512.
Twelve MVIVF saved-tree prediction-byte pairs match. The fresh installation
`/tmp/mvsic-flat-env` passes 63 tests and 42 subtests per ISA, the external CMake
consumer, all 160 public-ID cases and all 40 implicit-ID byte controls on each ISA. Older
environments remain preserved. Flat's applied worktree port is correctness-
validated but remains unstaged pending performance acceptance. VNNI is compile-
only; AMX is unimplemented. Production training defaults are unchanged.

The actual installed default-float block (`flat-installed-checkpoint-01`) uses
16 saved queries, no compression/reranking, one worker and forward/reverse order.
All sampled ID sets and score bits match, with recall 1.0 for both engines.
The strict serial gate nevertheless fails:

| Default float, ms | Original | Installed |
| --- | ---: | ---: |
| Mean | 162.555 | 163.597 |
| p95 | 166.400 | 167.787 |
| p99 | 167.005 | 168.487 |

These are 0.64%, 0.83% and 0.89% adverse differences. All timed endpoints were
CPU 112 on NUMA node 0, and candidate thread-user CPU was also higher; the records do not prove
a unique cause. The controller stopped without retries. The installed TQ1
profile and full 1,406-query quality checks were not run. Earlier prototype
successes and failed separate-process blocks remain separate evidence.

`flat-caller-cache-audit-01` investigates the float caller without another
timing run. Caching view geometry removes member-pointer reloads, but the
AVX-512 block caller still has 260 instructions and 1,087 bytes; all 96 common
Eigen text sections match exactly. The temporary candidate compiled once and
was not executed or applied. It does not establish a cause or remedy for the
small installed gap.

`flat-one-block-build-01` tests the separate one-document-block heap bypass.
The existing independent oracle and 504 added long-cloud result records pass
under both one and 16 workers. All four baseline/candidate and serial/batch
witness files match byte-for-byte; all 96 common Eigen text sections also match.
The one whole-query experiment preserves the prior preflight outputs, but mean,
p95 and p99 are 0.25%, 0.96% and 1.01% higher than the original. Absolute times
belong to a different CPU/NUMA block from the installed check. This does not
establish a useful speed gain or a unique regression cause. The candidate is
not applied, and further float-caller variants are stopped.

`mvivf-two-method-component-performance-01` compares the prior f1 refactor with
d52 using the same tree and all 1,406 queries. All eight prediction files and
actual DSO checks pass. Serial mean is 2.573 to 2.308 ms; median-repeat batch
throughput is 5,982 to 5,886 QPS (-1.61%), with overlapping process ranges.
Serial/batch peak-memory deltas are below 1 MiB. This is component validation,
not a fresh original-versus-refactor baseline, a build measurement or strict
overall nonregression. The earlier original-performance limitations below remain.
MPool/shared graph work is preparation-only.

## Earlier MVIVF-only selected-source correctness and packaging checkpoint

`public-id-final-validation-01` archives source
`f1ba2caaf9a5007a1343c1ebcf430caee03a615a22faf94bf5e8781a89dacd5c`.
All six native suites pass under Clang/GCC on AVX2/AVX-512; twelve strict
saved-tree before/current prediction-byte pairs match. The fresh installed
wheel passes48 tests and36 subtests per ISA, the external CMake consumer passes
both, all160 custom-ID cases pass per ISA, and all40 implicit-ID/score-byte
controls remain unchanged. Four previous installed environments are verified
unchanged; the new environment is `/tmp/mvsic-public-id-final-env`.

This includes the construction-selected custom-ID subclass and corrected
benchmark thread policy, with no YAML/CLI schema change. VNNI is compile-only;
AMX remains unimplemented. Production training defaults are unchanged.

The post-selection codegen audit compares all three ISA backends. Seven numeric
methods and eleven matched helpers have identical normalized instructions and
calls before/after; referenced load bytes, relocations, switch/string/TLS targets
also match for those methods. Link addresses/cacheline placement changed.
Thus no added hot operation was found to explain the small batch point estimate.
NUMA snapshots show placement variation but lack enough CPU tags for causal
attribution. Preserve that shared-host performance uncertainty below.

## Public-ID correctness update

Custom-ID cutoff ties were pruned using dense rows before public-ID mapping.
The fix retains public labels before pruning and resolves only selected rerank
candidates to rows. The reproducer covers all four codecs, single/grouped
search, no/float/TQ8 reranking, nonmonotonic large IDs, reorder and save/load:
120 of160 cases failed before; zero fail after on each host ISA. All40 implicit
ID/score-byte controls remain unchanged.

`public-id-validation-01` archives source hash
`a37e5d5080286544af3836801638906359d7d03e251398a1183d505550dd09d3`.
Six native suites pass under Clang/GCC on AVX2/AVX-512; twelve strict saved-tree
default prediction-byte pairs match. Installed tests pass45 tests and29 subtests
per ISA, with external CMake consumer checks on both. The installation is
`/tmp/mvsic-public-id-env`; older environments are preserved. VNNI is compile-
only; AMX remains unimplemented. No production training default changed.

The two implicit-ID serving blocks (`public-id-implicit-performance-01/02`,
ABBA then BAAB) retain byte-identical neighbors/scores. Across four processes
per implementation, mean/p95/p99 are2.162/2.677/2.860 ms before and
2.194/2.708/2.891 ms after. Individual before-process means range2.065–2.210 ms;
after range2.170–2.207 ms. One batch ABBA block gives6,737 versus6,539 QPS.
This shared-host evidence does not establish speed neutrality: retain the
small adverse differences and investigate the implicit path before acceptance.

The bounded same-process investigation `public-id-rerank-micro-01` uses eight
actual ArguAna queries and their selected coarse top50, with separate private
kernel/query lifetimes and one shared runtime. Moving the cold ID map after
the existing fields and capturing candidate data by value preserves every
rerank ID/score bit and byte counter. ABBA+BAAB medians are733.013→725.526 us
for TQ8 and905.918→892.788 us for float. Blocks remain noisy; these small
rerank-only gains do not establish whole-query performance. The minimal variant
is isolated and has not changed production source.

The follow-up `public-id-implicit-variant-performance-01` verifies the variant's
factory/vtable/core resolution and matches all prediction bytes, but fails the
whole-query gate. Across-process means are2.180932→2.272710 ms (+4.21%),
p95 2.706876→2.836125 ms and p99 2.970586→3.119642 ms. Both mean and median
aggregations, all eight processes and the earlier blocks are retained. Reject
this variant as a performance fix. A separate prepared candidate restores the
original implicit class layout/rerank body and isolates custom-ID translation
in a construction-selected subclass. That implementation is now selected as
described below; the rejected layout variant remains archived separately.

Selected kernel source SHA:
`9a7b9cf82202103b87bcf70605ffcfa7279e3def360d266c6daceca44cd220f0`.
`public-id-subclass-backends-01` first reconstructs all three old backend DSOs
from copied objects with byte-identical SHA256, then replaces only the MVIVF
kernel objects. No clustering translation unit or preload shim is involved.
All six Clang suites and ten installed API/custom-ID tests pass per host ISA,
with loaded paths verified. VNNI is compile-only. Full selected-source GCC/
wheel/installed validation remains separate.

`public-id-subclass-serial-01` matches every prediction byte on all 1,406 queries
and all eight processes, three repetitions each. Across-process median
mean/p95/p99 are2.680285/3.419853/3.771744 ms before and
2.331753/2.951330/3.281327 ms after. Across-process means are2.653588→2.330080 ms.
Before process means range2.351–2.903 ms and after2.270–2.387 ms: report a
shared-host nonregression gate, not a universal13% speed claim.

`public-id-subclass-batch-01` is one separate16-worker ABBA block, seven repeats
per process. Before QPS samples are5,997/5,693; after5,595/5,839. The aggregate
5,845→5,717 QPS is2.20% lower with overlapping process ranges. Predictions match
every serial/control byte, but strict batch acceptance fails this block. Keep
this result and the codegen investigation visible; do not claim overall speed
acceptance from the serial pass alone.

## November 12 IP-only API update

The public `Metric` type, metric properties/arguments and MVClustering
`inner_metric` selection are retired. Retrieval, outer assignment and inner
centroid training use IP/MaxSim. Private Euclidean k-means remains for quantizer
codebooks. The original YAML `metric: ip` remains accepted.

`ip-api-validation-01` archives source hash
`eb1b638cda15083d5e74131ac617f1370e62410fd09caf4151a3088bdbbd71bf` and passes
six CTest suites on AVX2 and AVX-512 under both Clang and GCC. Twelve strict
saved-tree previous/current prediction-byte pairs match, covering serial/batch,
root-M2M and compressed-query/TQ8-rerank paths. Each installed ISA run passes
41 tests and 29 subtests, including the original YAML workflow; the installed
CMake consumer passes both ISAs. `/tmp/mvsic-ip-api-env` is the new installation;
the two research environments are hash-verified unchanged. VNNI is compile-only;
AMX remains unimplemented. This is correctness/packaging evidence, not a new
runtime-speed claim. Shared-host serving limits below still apply.

Training experiments remain separate and have not changed production defaults.
See [TRAINING_EXPLORATION.md](TRAINING_EXPLORATION.md) for accepted/rejected
research branches and retained artifact locations.

## Measurements

`public-client-compile-cost-01` measures corresponding C++ client translation
units against the frozen original IP/TQ1-leaf/TQ4-center implementation and the
exact installed refactor headers. Both construct/configure the index and call
build/search/save/load with Clang15, O3, AVX-512, one compiler and no link.
Original MVIVF header hash matches the retained baseline source manifest.

| Client compilation | Original | Refactor |
|---|---:|---:|
| Wall seconds | 103.94 | .87 |
| CPU seconds | 103.92 | .87 |
| Compiler peak RSS KiB | 1,181,900 | 138,224 |
| Object bytes | 8,012,008 | 11,968 |

The about119x client compile reduction is a concrete benefit of compiled
implementations and the small public API. It excludes linking and the shared
library's own compilation; it does not complete the clean whole-library cost
comparison listed below. One run per translation unit is retained with source,
compiler command/dependency hashes and required external-symbol checks.

ArguAna corpus: 8,674 documents, 1,208,012 vectors, dimension128. Retrieval uses
TQ4 centers, TQ1 leaves, TQ8 rerank50, Wards0.5, k10. Training defaults remain
float/IP. Absolute timings from different host-load conditions must not be pooled.

| Measurement | Original | Refactor | Evidence scope |
|---|---:|---:|---|
| Float index build, median seconds | 38.173 | 37.236 | `arguana-leaf100-final-01`, quiet host, 8 workers, two builds/engine, before latest Parlay changes |
| Build peak RSS, median MiB | 1687.8 | 1089.0 | Same builds; 35.5% reduction |
| Single-query mean latency, ms | 2.230 | 2.129 | `parlay-recovery-queue-01/comparison`, latest source, shared host with disjoint affinity |
| Single-query p50 / p95 / p99, ms | 2.172 / 2.753 / 2.974 | 2.077 / 2.617 / 2.815 | Medians of selected-repeat percentiles across two processes; full1406 queries |
| Single-query peak RSS, MiB | 1483.5 | 847.9 | Same latest-source comparison |
| 16-worker batch QPS, first comparison | 6466.7 | 6365.3 | Same shared-host run; new -1.57% |
| 16-worker batch QPS, immediate rerun | 6190.1 | 6294.5 | `immediate-compare16-01`, user-authorized busy host, unpinned, new +1.69% |
| Immediate batch time, ms | 165.4 | 162.8 | 1024 queries; balanced ABBA, 20 repetitions/process |
| Immediate batch peak RSS, MiB | 1532.3 | 926.5 | 39.5% reduction |

QPS summaries use the median of each process's fastest repetition, matching the
existing comparison harness. Both latest comparisons use two processes per
implementation/workload. Busy/shared-host results support approximate parity at
16 workers, not a rigorous speed-win claim.

Optional assignment optimizations separately reduced TQ4 whole-build time from
82.989 to38.664 seconds (2.146x), and TQ1 from131.505 to31.005 (4.241x), with exact
tree hashes and all measured recalls unchanged within each mode. These are
ArguAna assignment-mode studies, not the default float-build comparison or a
demonstration of sub10-second FiQA construction.

## Correctness and repeatability

The earlier Parlay recovery source is archived in
`parlay-recovery-queue-01/build-validation/native-source.zip`. All six native
suites pass with Clang under AVX2 and AVX-512, including scheduler sharing,
one-worker behavior, deterministic boundary error propagation, all serving
codecs and owning-query lifetime tests. Latest/current refactor prediction files
are byte-identical in all retained workload comparisons.

`work-trace16-immediate-01` compares original and the preceding final-build-04
snapshot on all1024 shared-tree queries. Chosen leaves and shared coarse float
score bits match. Both engines make139108 center and487116 leaf cloud comparisons
(626224 total), with equal center-vector and leaf-vector totals. All398 differing
rerank shortlists exchange exact coarse-score ties. Selected rerank-vector work
differs by only0.00681%; twelve final neighbor sets differ in that trace.

Latest latency recall is .908534851 original versus .908819346 refactor.
Immediate batch recall is .9078125 versus .9080078125. These use different query
sets and must not be compared across modes. Original repeat-to-repeat tie
membership can vary; current refactor outputs are stable in retained trials.
Freshly built trees can also differ because original initialization/padding bugs
were fixed; those curves are recorded separately in the leaf100 study.

## Unfinished acceptance requirements

- Latest Parlay recovery has not been remeasured at96/192 threads. Earlier
  validated snapshots were about5–7% slower there, so full-machine parity has
  not yet been demonstrated and the earlier regression must remain visible.
- A latest-source default construction pair now completes as reported above;
  its build/quality point estimates do not establish unconditional parity.
- Applied d52 source passes the refreshed installed-wheel/C++-consumer and
  Clang/GCC checks above; performance acceptance remains separate.
- Clean original/refactor compilation-cost comparison is not complete. The
  latest incremental native build took162.70 seconds with four compiler jobs;
  it is not a clean-build acceptance measurement.
- VNNI remains compile-only; AMX is not implemented or hardware-validated.

The existing MVIVF checkpoint and unstaged Flat worktree are reviewable work in
progress. They must not be described as having satisfied the user's unconditional "at least as good as
before" criterion until the remaining evidence is collected.
