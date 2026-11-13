# MVIVF refactor handoff — branch `reformat`

## Active work resumed

The final-root-only TQ1 top16 plus Float16 screen passes its fixed-state gate,
but its one whole FiQA candidate is **not promoted**. The44.771-second build
retains byte-identical Float C5 centers and changes37/57,638 final root labels.
Full648-query p32 hits improve3711→3729 with identical shared-ID score bits.
Balanced serial mean/p95/p99 regress0.94%/1.52%/2.23%, and median full-sweep
time regresses1.23%; all four local pairs are slower. No batch, alternate probe
or retry. Evidence: `final-root-float16-screen-01` and
`root-final-float16-build-01`. The subsequent genuine-C1 TQ8 top8 plus Float
screen matches every4096 Float label and score bit but fails cost:1.035914 s
versus Float.714573 s. No whole build or grid follows. Exact Float query batching
also fails cost: after preserving the original single-center kernel, all12
fixtures and4096 C1 labels/score bits match, but streaming is16.48% slower.
Existing AllPairs gains only4.35%, below10%. Both batching attempts are archived
under `exact-batched-float-c1-screen-{01,02}`; no full build or production change.
These branches are closed. Active next phase is raw SVHIVF: freeze the original
typed oracle and v3 fixture, then port the concrete API and prove output parity
before fresh-build or performance runs. Scratch preparation is under
`/tmp/mvsic_svh_raw_oracle` and `/tmp/mvsic_svh_raw_port`.
The six-file source78d7 checkpoint below remains staged and unchanged.

The user pointed out that the validated low-recall TQ1 checkpoint was not the
end of the optimization task. Continue toward Float-quality build speed.
Two exact screens pass: borrowed rows reduce a FiQA Float pair49.101→45.761 s
with23.29% lower peak, and root-query reuse reduces a TQ1 pair28.107→25.867 s,
avoiding21.98M inner encodings. Trees remain exactly e174fc/dad381 respectively.
The five-file patch plus two existing-suite regressions and the Parlay grouping
change below are now staged. Normal selected-source Clang/GCC builds, both host ISAs, installed
Python/YAML/C++ checks and both combined exact-tree gates pass. Source 65ed65ad
(348 files plus three packaging READMEs), environment
`/tmp/mvsic-selected-view-cache-env`, wheel a158c659; no unaccepted Flat port is
compiled or packaged. Its batch regressions (−3.35% versus original, −1.63%
versus previous current runtime) remain archived. Alignment screens are closed:
panel alignment misses its gain threshold; function alignment64 worsens latency.

A current-core probe identifies serial request construction at 1.924 ms. The
original used flat Parlay grouping. That path is restored in one additional
existing file, mvivf_search.cc, preserving query/group8/scatter order. Source
78d7c073 passes correctness and its declared serving gate: median batch QPS
15,010 versus original 15,005, up5.09% versus source65ed; serial mean/p95/p99
improve3.92%/2.74%/2.56% versus original at higher recall. Mean batch QPS is
still1.86% lower than original, and serial mean is1.93% slower than source65ed.
All current outputs retain exact1fff7a92 bytes. Final GCC/incremental-wheel/
installed checks pass: 24 native suite invocations, 12 exact predictions,
48 tests plus36 subtests per host ISA, and both consumers. Current environment
`/tmp/mvsic-parlay-requests-env`, wheel5a1eff7d. Only the facade plus RECORD
changes from the prior normal wheel; executable/relocation/layout proof passes.
The six verified files and reports are staged. All348 selected source blobs
plus three packaging READMEs match compiled78d7; unaccepted Flat stays unstaged.
No commit or push. No blanket
all-metric speedup or additive independent build-gain claim is justified.

The original FiQA Float reference completes in57.064 s. Common-runtime full648
p32 recall is56.913580% original/57.268519% current Float. A private new outer
policy builds in40.925 s with57.978395% recall: Float first/final, intermediate
TQ1 top8 plus Float refinement including mapped genuine previous centers, and
Float legacy10 inner training. Same-p32 median serial latency is1.20% worse.
The sole matched-recall p31 point retains3718hits versus Float p32's3711,
improves mean/median, but worsens p99 by2.93%; the strict gate fails. No batch,
repeat, alternate probe or public/default change is accepted. A no-preload Python symbol-audit
preflight failure is preserved separately; the corrected RTLD_NOLOAD ownership
check passes without loading/promoting any library, and actual quality passes.
No further graph grid or representative-size tuning. See REFORMAT_PLAN.md and
the SSD artifacts for current gates. Keep working beyond the checkpoint.

## Previous exact query-load checkpoint

The private TQ1 query-word copy optimization is applied and staged after complete
validation. FiQA TQ1 at192 workers improves35.494→24.352 s with the exact same
224,221,500-byte index (SHA dad38138). CPU falls6310→4194 s; peak changes+0.085%.
This preserves existing TQ1 quality, including its deficit versus Float. Float
remains default; the six-second result below is ArguAna, not FiQA.

Installation for that checkpoint `/tmp/mvsic-queryword-env`, validated worktree
source3028d477, selected staged inventory89ee99ff. All Clang/GCC native suites,
twelve saved-tree prediction-byte cases, installed63 tests plus42 subtests per
host ISA and external consumers pass. Only changed clustering objects were
rebuilt; the wheel transparently replaces three ISA libraries plus RECORD,
with unchanged payload/ELF/loader proofs. VNNI is compile/ELF-only. Evidence:
SSD `tq1-queryword-{probe,override,boundary,validation}-01` and
`staged-queryword-checkpoint-audit-01`. No commit or push.

The genuine Float C1 warm-start screen is complete, using C0 labels remapped
after group compaction. Two empty clusters move32 occupied center positions.
On4096 query positions, previous-plus-entry0 raises ANN Float4 assignment
agreement81.52%→95.61%; dense TQ1+Float4 reaches97.58%. It reduces scored centers
but fails the declared quality and total-cost gates. Query-only speed improves
3.32x; graph/cache rebuild costs0.138 s on this subset. No full-tree ANN/default
change is accepted. Offline top4-union-previous Float refinement predicts
96.46% warm versus98.63% dense agreement, without a new native/timing run.
Artifacts: `outer-assignment-warm-c1-4096-01`. All training rows, duplicates and
weights remain unchanged; saved current-state labels never seed search.

Encoding already uses FhtKacRotator, also found in local MV-PiPNN. The private
`tq1-encoding-profile-01` probe now measures one-time root preparation1.008 s:
descriptor storage.297 s and bundled encoding.693 s. Root training/final
assignment envelopes total10.168 s, of which10.076 s (99.09%) is scoring and
.013 s center encoding; query codes are reused. This isolates clustering root
assignment, not every encoder in the whole index build. A source audit proves TQ1 inner queries can reuse
immutable root encodings with the existing TQ8-style root-vector mapping.
Queries currently encode once per inner call, then reuse those codes across
Lloyd steps; changing centers still require fresh encoding. Audit/proposed
measurement: `/tmp/mvsic_tq1_encoding_audit`. Host perf sampling is unavailable
(`perf_event_paranoid=4`, no profiling capability); one permission probe is
preserved, no retry. The phase probe passes MC19, exact dad381 tree and actual
library/factory guards. Inner work re-encodes21.98M query vectors and18.25M center
vectors; these overlapping task walls cannot establish CPU or end-to-end shares.
The diagnostic build time is not a speed comparison. No encoding rewrite is
applied. Source proposal: `/tmp/mvsic_tq1_encoding_audit`.

## Latest applied worktree checkpoint (November 12, 2026)

Current user direction: retain automatic representative sizing (`s=0`), close
fixed-size/ratio/per-level tuning, and use 192 workers only for necessary tests.
The previous opt-in fixed64 YAML examples have been removed from this checkpoint.
Historical size experiments remain archived, not recommended as next work.

The corrected MVIVF checkpoint now passes the tested 192-worker acceptance gate.
Fresh ArguAna Float build is6.890→6.387 s, peak RSS3,570→3,021 MiB. With each fresh model
at p32, serial mean/p95/p99 improve1.56%/0.15%/0.94%; median batch QPS improves
1.96%, and current recall.943954 exceeds maximum original serial recall.940967.
The predeclared serial gate passes before batch runs. Shared-tree serving also
passes, with mean latency1.834→1.816 ms and median batch QPS19,088→19,579.
Both studies retain every sample and actual installed-library identity.

Current outputs are byte-identical across modes/processes. The learned Float
tree retains SHA62487820 across earlier16/192-worker builds. Fresh original
trees differ; original zero padding/reseed issues remain, and new p16 build
recall is lower while p32 is higher. Do not imply dominance at every budget.
See `tests/refactor/CHECKPOINT_REPORT.md` and SSD reports
`mvivf-default-build-pair192-01`, `mvivf-own-tree-p32-serving192-01`, and
`mvivf-wrapper-installed-serving192-01`. Flat remains separately unaccepted.

Current FiQA research reference is complete: Float/TQ1 builds take 45.880/35.618 s
with the same diagnostic proxy, but p32 recall is .572685/.534568 and TQ1 uses
more memory. Coupled quantization remains experimental. Float root assignment
takes 18.652 s versus 2.220 s updating across five measured iterations; final
assignment remains in the untimed residual. The next ANN screen reuses that
saved Float root and exact partition labels with 4,096 sampled queries, avoiding
a new training/capture compile. Native fixtures precede the real data screen;
the first native pilot passes fixtures and all 4,096 saved Float labels, but its
stock R8/L16/alpha1 graph stops before queries. A same-binary debugger check
proves zero cache misses and only 196/241 representatives reachable from entry 0.
This configuration is parked; no ANN quality/speed result or default change is
accepted. Next is the bounded exact TQ1 query-word copy microbenchmark, with
all padding/numerical/sanitizer gates before any production edit.

Highest priority is a confirmed shared-runtime wake-up defect. Sharing Parlay's
scheduler TLS is insufficient: `-Bsymbolic` gives each ISA library its own
libstdc++ atomic-wait fallback table. Notifications from another library can
miss sleeping workers. `waiter-runtime-build-01` reproduces this with the same
compiled owner object: the old link uses one of four workers for all 32 jobs;
a narrow diagnostic relink shares the table and uses all four. Both complete
all jobs and share scheduler TLS. All three actual backends have been relinked
without recompilation; only the waiter-table relocation changes. VNNI receives
static ELF checks only. Real build/output gates are now recorded above.

Permanent fix now applied and validated: three scheduler wait/notify wrappers in
the existing shared core, with ISA `-Bsymbolic` retained. The new idle-pool test
passes under Clang/GCC and fails against the exact old runtime. The complete
`wrapper-timer-validation-01` checkpoint passes seven suites on both host ISAs
under both compilers, twelve saved-tree prediction-byte cases, installed 63 tests
plus 42 subtests per ISA, original YAML and the external C++ consumer. Current
environment: `/tmp/mvsic-runtime-checkpoint-env`. Compiled source is04b9a413;
a separately hashed README packaging supplement leaves every code/binary payload
unchanged. VNNI is compiled/ELF-checked only. Installed performance gates above pass.

The permanent fix routes scheduler wait/notify through the shared core,
retaining private codec/Eigen bindings and avoiding a libstdc++ internal-symbol
dependency. Do not promote borrowed rows or infer a clean copy-only speedup:
its preload also interposes the waiter table. A per-node diagnostic observed
44 baseline clustering calls on one thread versus43 threads with that preload,
despite the same192-worker pool and exact tree bytes. This is a runtime
confound, not evidence that copying alone changes descendant parallelism.

`mvivf-serving192-01` completes serial1 ABBA then batch192 ABBA on the prior
shared-original tree. Serial mean/p95/p99 improve 1.876/2.332/2.572 to
1.768/2.172/2.351 ms. Median-repeat QPS is 19,308 to 19,175 (-0.69%), but
mean-repeat QPS is 19,136 to 16,248 (-15.09%). Both refactor first timed batches
take 113–115 ms versus 72–74 ms subsequently despite one full warmup. Preserve
the adverse samples. Current predictions are byte-identical across all four
serial/batch processes; shared-ID score bits match the original.

The single caller-only diagnostic `mvivf-batch192-phase-trace-01` reproduces that
transient and localizes about38 ms to query preparation: 40.197 versus
2.212/1.799 ms later, with12,874 versus1,562/952 minor faults. Root/leaf scoring,
reranking and destruction have no comparable excess. Correlation is not a
causal allocator/fault attribution. No numeric kernel was rebuilt or changed.
The trace library also has its own local waiter table, so its phase observations
cannot establish the cause of the uninstrumented transient. Verify the eventual
runtime fix without a profiling preload.

FiQA AUTO float reaches a180-second worker cap at192 without an index
(`fiqa-auto192-01`); the paired controller stops before TQ1. One separately
declared TQ1 stage diagnostic reaches its60-second cap, also without an index.
Its root five iterations report11.66587 s assigning and14.27718 s on the old
inclusive update clock. A timer-accounting bug double-counted assignment:
exclusive updates are about2.61131 s, and measured iteration phases total
about14.27718 s, not25.94305 s. The reset fix passes all 19 clustering tests on
AVX2 and AVX-512; the same regression fails against the old timer as expected
(`clustering-timer-validation-01`). The current checkpoint environment contains
the corrected timer; earlier environments retain the old timer. No completed FiQA
build/quality or float-relative speed
ratio can be inferred. Actual worker192 threads/all192 affinity and unrestricted
CPU quota were checked; representative-size log lines do not establish
descendant serialization.

The exact borrowed-row prototype (`borrowed-rows-build-01` and
`borrowed-rows-arguana-hash-01`) passes18 MVClustering tests, the zero-copy/order
oracle and exact Float/TQ1 full-tree hashes. One192-worker pair reduces Float
wall/CPU/peak by 10.0%/9.4%/37.2%. TQ1 CPU/peak drop5.8%/35.0%, while wall rises
18.2%; retain that adverse point. It remains temporary and unpromoted.
Its performance attribution is suspended by the wake-up defect above.
Outer ANN over multivector representatives is a distinct, worthwhile next
screen; rejected cheap inner-vector ANN settings do not settle it. No outer
graph implementation or experiment has run yet.

`mvivf-default-build-pair16-01` completes one fresh original/installed default
ArguAna build pair at 16 workers. End-to-end build/preparation is 22.058 versus
22.472 s (+1.88%); GNU-time peak RSS is 1,980.73 versus 1,431.19 MiB (-27.74%).
Both trees load correctly. Common-runtime recall on all 1,406 queries is
.910740 versus .909175 at 16 probes, and .944452 versus .943954 at 32 probes:
22 and 7 fewer correct neighbors out of 14,060. This closes the missing fresh
build measurement, but does not establish unconditional build/quality parity.

The installed tree is byte-identical to both earlier refactor builds; the new
original tree differs from its earlier builds. Offline inspection proves one
specific original padding defect in this pair: root center 59 represents the
same 19-vector document 7264 in both trees. Original stores those 19 vectors
followed by 121 zero vectors; the refactor repeats the real vectors throughout
the 140 slots. This preserves that document's directed MaxSim semantics.
Original saved centers contain 468 nonunit vectors, with norms from 0 to 5.28;
all refactor centers are unit within 1.17e-7. This is real training-content
change, not serving/timer noise. It does not apportion the whole recall delta;
the earlier unsynchronized original reseed counters remain a separate source
defect, without per-center causal attribution. No source is changed to reproduce
the original invalid padding. See the new pair's report and tree audit.

The Flat integration is applied and correctness-validated in the worktree;
its performance acceptance and staging remain pending. `flat-integration-validation-01`
archives source `d52f30771a654738fcf1e4ad101203571a91ab65dd4b8a9538065a899a0e0c8f`.
All seven native suites pass under Clang and GCC on AVX2 and AVX-512. Twelve
MVIVF saved-tree prediction-byte pairs match. The installed wheel passes 63 tests
and 42 subtests per ISA, the external CMake consumer passes, all 160 public-ID
cases pass and all 40 implicit-ID byte controls match on each host ISA. The installation
is `/tmp/mvsic-flat-env`; older environments remain preserved. VNNI is compile-only;
AMX is unimplemented. Training defaults are unchanged.

`flat-installed-checkpoint-01` tests the actual installed libraries. Default
float search without compression or reranking preserves all sampled IDs/score
bits and recall, but fails the strict serial gate: mean, p95 and p99 are 0.64%,
0.83% and 0.89% slower than the original. The planned TQ1 profile and full-query batch
quality checks were not run. Flat remains unstaged pending performance review;
earlier prototype wins do not replace this installed result. MPool/shared graph
work remains preparation-only under `/tmp/mvsic_mpool_migration`.

The isolated one-document heap bypass passes scalar/codec/ownership and exact
serial/batch witnesses, but its one whole-query mean/p95/p99 ratios are
1.0025/1.0096/1.0101 against the original. It is not applied. Both Flat caller
variants are now stopped; evidence is in `flat-one-block-build-01` and
`flat-caller-cache-audit-01`.

`mvivf-two-method-component-performance-01` compares the prior f1 refactor with
the d52 two-method runtime, using one shared tree and all 1,406 queries. Prediction
bytes match exactly. Serial mean is 2.573 to 2.308 ms; batch throughput is 5,982
to 5,886 QPS (-1.61%), with overlapping process ranges and peak-memory deltas
below 1 MiB. This is component validation, not a new comparison with the original
MVIVF. Earlier original-baseline and full-machine performance limits still apply.
The task is not finished and unconditional speed parity is not established.

## Earlier MVIVF-only selected-source validation (November 12, 2026)

`public-id-final-validation-01` validates source
`f1ba2caaf9a5007a1343c1ebcf430caee03a615a22faf94bf5e8781a89dacd5c`:
six native suites × Clang/GCC × AVX2/AVX-512, twelve saved-tree byte pairs,
48 installed tests plus36 subtests per ISA, external CMake consumer and the
160-case public-ID/40-case implicit-byte reproductions all pass. New installed
environment `/tmp/mvsic-public-id-final-env`; four older environments preserved.
The custom-ID subclass and benchmark thread-policy fix are included. VNNI is
compile-only; AMX unimplemented. Keep runtime acceptance scoped to the recorded
shared-host studies and the adverse batch point estimate below.

## Public-ID correctness and method migration (November 12, 2026)

Custom corpus IDs exposed a real cutoff-tie bug: pruning by dense row before
translating the surviving neighbors could discard the smallest public ID.
Prepared leaves now retain public labels; custom-ID float leaves are ordered
once before local-position pruning, and only selected rerank IDs are resolved
back to dense rows. Implicit-ID scoring/arithmetic is unchanged.
`public-id-validation-01` passes the six native suites under Clang/GCC on both
host ISAs, twelve strict default prediction-byte pairs, 45 installed tests and
29 subtests per ISA, and the external installed CMake consumer. The 160-case
public-ID reproducer goes from120 failures to zero on both ISAs; all40 implicit
ID/score-byte controls stay unchanged. Installation: `/tmp/mvsic-public-id-env`.
The prior research/API environments are preserved. Correctness is verified;
the separate shared-host timing investigation is still open.

The document-scaled/aligned half-average center policy improves ArguAna, but
fails the short-cloud Quora latency gate. It is not a global default. The
existing automatic-average policy remains. The Quora fixture is an exact
whole-document subset with proven top10 completeness, not full-corpus recall.
See `tests/refactor/TRAINING_EXPLORATION.md` and `REFORMAT_PLAN.md`.

Further bounded screens: Wards0.5 post-training center compression fails quality
and stops without serving expansion (`arguana-tree-compression-gated-01`). FiQA
fixed64 completes in81.861 s at16 workers and improves full648-query recall/
serial latency against a retained historical tree (`fiqa-size64-01`). That old
training provenance is imperfect; a fresh same-source default control is needed
before attributing the gain to size alone. No192-thread measurement is implied.

The tiny ID-map-layout variant fails whole-query latency despite its micro gain
and is rejected. The selected subclass implementation restores the original
implicit kernel layout/rerank body and adapts custom IDs only at construction/
rerank boundaries. Source SHA is9a7b9cf82202103b87bcf70605ffcfa7279e3def360d266c6daceca44cd220f0.
All six Clang suites and targeted installed API tests pass on both host ISAs;
normal-library serial ABBA+BAAB matches every output byte and passes latency.
The one batch block is2.2% slower with overlapping process ranges; keep that
adverse result visible. Selected-source wheel/GCC validation has now passed.

Installed benchmark thread policy was corrected: configured workers control
Parlay; other library limits default to1 and preserve explicit inherited values
outside serial latency. Serial latency forces all limits to1. Both YAML paths
record provided/effective limits and fingerprint build/search separately.
Focused tests pass27 tests plus36 subtests; exact YAML/CLI schemas are unchanged.

Flat's early isolated build/oracles passed (`flat-float-build-01`), while its
first balanced serial block failed (`flat-float-confirm-01`). Those historical
results remain visible. The later applied Flat integration now passes native,
installed API/YAML and consumer correctness checks, but remains unstaged after
the installed default-float latency failure above. MPool/shared graph work is
still preparation-only. A method is not accepted merely because its port is
applied. Preserve method-specific stats: tuple field2 estimates database bytes;
the historical YAML `avg_cmps` column is a compatibility name.

## IP-only API cleanup (November 12, 2026)

The user retired L2 from MVClustering after the inner-objective pilot and the
four-cell outer/inner screen found no MVIVF benefit. The public `Metric` enum,
metric arguments/properties, and `MVClusteringConfig.inner_metric` are removed.
Use `IndexMVIVF(d, params)` and `cluster_point_clouds(points, k, s, config)`.
Both outer assignment and inner centroid training use IP, with unchanged float
updates, document weights and spherical normalization. Private L2 k-means and
shared Euclidean quantizer math remain. Original YAML `metric: ip` is preserved;
Python/config boundaries reject L2 clearly. `ip-api-validation-01` passes all
six native suites on AVX2/AVX-512 under Clang and GCC, twelve strict saved-tree
prediction-byte pairs, 41 installed API/YAML tests plus 29 subtests on each ISA,
and an external installed CMake consumer. The isolated installed environment is
`/tmp/mvsic-ip-api-env`; both older research environments remain unchanged.
VNNI is compile-only and AMX remains unimplemented.

Research is tracked in `tests/refactor/TRAINING_EXPLORATION.md`. Quantized
top-2 float refinement is promising; TQ4/TQ8 refinement, the tested ANNS
settings, L2 and additional EMA variants are paused on the recorded evidence.
Fixed64 on ArguAna wins matched/higher-recall serial latency by about 12% on
two seeds, but is not a universal default. Current priority is budgets relative
to document cloud lengths on ArguAna and short-cloud Quora; center-only vector
compression is prepared but parked. Experimental sources/results are on SSD;
production training defaults have not changed. Check the latest session/run
status before launching another native process.

## Immediate 16-thread work trace completed

The user explicitly authorized immediate 16-thread execution on the busy host,
without waiting for idle CPUs or pinning worker affinity. The old waiting queue
was superseded. `refactor/work-trace16-immediate-01` compiled the temporary
diagnostic and completed original/final-build-04 batch traces at 16 workers.
CPU-load monitoring was disabled by explicit request; timings are excluded from
performance claims. The EMD coordination marker remained inactive.

The work-count gap is now closed on all 1,024 queries of the shared pilot tree:
both engines expand 139,108 center clouds and score 487,116 leaf documents
(626,224 total logical comparisons). Center-vector totals (19,514,674),
leaf-vector totals (67,561,774), effective query embeddings (13,088), chosen
leaf groups and shared coarse score bits match exactly. All 398 differing
rerank-cutoff sets replace candidates with identical coarse-score-bit multisets.
Rerank vector work is 7,487,621 original versus 7,488,131 refactor, only 0.00681%
more. Twelve final neighbor sets differ in this trial. The traversal/work
discrepancy cannot explain a 5% serving gap; candidate identity changes at coarse
ties remain relevant to quality. Detailed per-query arrays, source/runtime
identities and `work-analysis.json` are retained in that directory.

## Parlay recovery checked; remaining work trace queued

Latest scheduler decision: the user recalled the historical change and explicitly
chose to keep the one-worker scheduler for latency. Do not add a separate
compile-sequential package or latency path. Git history confirms `bench_search_seq` originally
defined `PARLAY_SEQUENTIAL`; commit
`300e67e28ebff782e330bc247aae689f8a67e1e4` (May 4, 2026, "seq search fixes;
label fixes") removed it when splitting benchmark instantiations and replaced
it with the one-worker scheduling policy. The user's recollection is correct.
Verification of the current original workflow found two routes:
native `bench_seq` / global `--config=serial` compile sequentially, while the
current original YAML latency runner calls `compute_stats_latency` on its
installed extension and uses `execute_with_scheduler(1)`. That Python route can
also be compile-sequential if its entire extension was built with the serial
config; source scripts do not prove historical wheel flags. Current validation
uses the runtime one-worker path and does not claim compile-sequential performance.

Latest user direction: use at most 16 Parlay workers on physical cores disjoint
from active external jobs, including their SMT siblings. Do not use all 192
threads for the next tests. Preserve single-query latency and restore batch
throughput before pursuing training experiments.

The pinned Parlay source and original MVIVF were reviewed before these changes:
prepared query owners and nested probe vectors are now normally constructed
Parlay sequences, restoring parallel destruction. Batches of at least 32 queries
validate independently on workers, catching all callback exceptions and retaining
the lowest-row exception under a failure-only mutex. Structural checks keep their
original precedence; errors are rethrown on the caller before preparation.
Tests cover valid grouped tails across all four codecs and first-error precedence
with one/four workers. These changes preserve the current traversal, arithmetic,
candidate layout and tie rules.

The private Parlay overlay additionally runs plain loops and left/right calls
when its scheduler has one worker, before adaptive-granularity clocks or job
publication. This is a runtime serial fast path, not a `PARLAY_SEQUENTIAL` build.
Original `bench_seq` and `--config=serial` use compile-time sequential scheduling;
the newer original `bench_search_seq` and Python latency use a one-worker pool,
as the installed refactor currently does. No extra ISA libraries or public API
were added for the fast path.

The recovery queue completed on 16 disjoint physical cores (CPUs 0–15), with four
compiler jobs. All six native suites pass on AVX2/AVX-512 under Clang, and all
12 comparison processes completed with byte-identical current/candidate outputs.
Results are under `refactor/parlay-recovery-queue-01`; its native source archive
and compile commands are in `build-validation`. The installed final-wheel-03
still contains the preceding validated implementation; wheel refresh and GCC
revalidation remain pending. Queue status is recorded at
`/ssd2/kishen/mvsic_new/refactor/parlay-recovery-queue-01/status.json`.
Shared-cache/memory contention remains possible even with disjoint CPU affinity;
these results cannot replace full-host performance acceptance.
A follow-up queue at `refactor/parlay-work-trace-queue-01/status.json` was blocked
before compilation when those cores became busy again. Its replacement is
`refactor/work-trace16-queue-01/status.json`, waiting for EXACTLY 16 disjoint
physical cores plus their SMT siblings to stay available for 30 seconds. It
retries only guard-blocked/interrupted attempts, preserving separate directories;
build or correctness failures stop it. It runs four temporary original/
final-build-04 work traces (one worker and 16 workers).
It records expanded center-cloud/vector counts, selected leaf document groups,
leaf-vector work and actual rerank-50 IDs/coarse score bits. Instrumented timings
are excluded from performance claims; root-M2M logical counts are subsets of
center totals. This closes the earlier counter gap without changing the public
API. Generated sources, scripts and runtime identities are archived.

## Read-only serving audit checkpoint (November 11, 2026)

The queued 192-thread serving study completed at 18:32 Eastern on its first
attempt with no interference. Median throughput across all 25 repetitions is
18,897 original / 17,590 prior refactor / 17,951 current refactor queries/s.
Current and prior refactor predictions are byte-identical. Last-ten timing
summaries retain the regression, so warmup does not explain it.

The source audit in `tests/refactor/SERVING_AUDIT.md` finds measured serial
validation (about 3 ms at 96 workers) and serial prepared-query destruction
(about 2 ms) as the strongest targets. The original uses parallel Parlay
destruction. Saved root/leaf/rerank stage timings do not show the scoring
arithmetic getting slower. No production changes or new native jobs were made
for this audit. Keep performance acceptance open until isolated fixes are tested.

Historical full-coarse outputs match visited IDs and float score bits on all
128 queries on each of two fixed trees. The full 1,024-query comparison still
lacks new-engine batch center/leaf counters; original total is 626,224 throughout
its retained 192-thread trials. Do not claim full work equivalence from the
old refactor-only cross-tree stats. Original tie rules and corrected padding
also preclude blanket output equivalence on arbitrary inputs. Independent
training changes remain separate from serving-only performance fixes.

## Current validation and measurement checkpoint (November 11, 2026)

The latest source provisionally adds TQ1 byte-popcount accumulation on AVX2 and
AVX-512 without VPOPCNTDQ, selected once for 2–31 tiles. Its unchanged one-tile
and larger-tile paths retain the prior accumulator. Two AVX-512 cross-query
totals now use int64: the actual long-cloud regression demonstrates that the
old int32 total returned approximately -3 instead of the correct +1. A second
build of the existing panel test source directly exercises AVX-512 helpers,
with a baseline-ISA entry point checking CPU/OS support. No new `.cc` file was
added. `final-build-04` archives the candidate; all six suites pass on both host
ISAs with Clang (`final-validation-03`) and GCC (`gcc-final-validation-02`).
The installed `final-wheel-03`, external consumer `cpp-install-05` and all 84
strict checks in `final-parity-03` pass. The broader historical measurements
below use `final-build-03`. A 44-process complete-query comparison is now running
against the original and that prior refactor; do not treat the candidate as
accepted until those comparisons pass.

Temporary exact-oracle studies `tq1-byte-accum-001` and `avx2-hamming-002`
support the candidate. At dimension128, AVX-512 scoring plus query setup
improved 1.125x for single and 1.157x for grouped queries; the d1024 fallback
was approximately unchanged. AVX2's gated candidate improved the prepared IP
helper about 1.10x at d128; the one-tile path is approximately unchanged.
The first AVX2 prototype regressed at one tile, so that route was excluded.
These are kernel studies, not complete-query speed claims. The separate packed
selection experiment (`selection-001`) passed its ordering oracle but saved
only about 0.74–1.10 microseconds on retained score distributions with timing
sensitivity; it was not adopted.

This section supersedes the older checkpoints below. Implementation and installed
packaging pass the suites listed below; the expanded top-k tie regression has
been fixed and passes the broader serving study. Performance
acceptance remains in progress. Do not
commit or push. Keep the original repository read-only.

- The combined native source is archived and hash-verified in
  `refactor/final-build-03/native-source.zip` and its manifest. All five native
  suites pass on both AVX2 and AVX-512 with Clang (`final-validation-02`) and GCC
  (`gcc-final-validation-01`). GoogleTest covers
  retrieval, ownership, persistence, concurrent first-use TQ8 reranking,
  clustering and dispatch. The numeric boundary test checks 11,460 fixtures.
- The actual installed wheel passes all eleven API/CLI tests outside the source
  tree (`final-wheel-02`). The installed CMake package passes an external consumer
  (`cpp-install-04`). All 84 final strict parity checks pass in
  `final-parity-02` (AVX2 and AVX-512); the original AVX2
  single-query implementation is the correctness reference. All new batch/root
  modes remain checked. The latest lightweight suite passes 57 tests and 29
  subtests (`lightweight-final-01`).
- `/tmp/mvsic-final-env` contains the final wheel. Preserve the earlier
  `/tmp/mvsic-refactor-env` installation for quantization baseline measurements.
  The wheel build leaves CMake testing/comparison options OFF; restore them
  before rebuilding those targets. Existing native executables remain usable.
- TQ4 inner assignment now packs moving centers for both supported ISAs. Clean
  `inner-assignment-004/005` runs measured approximately 16.3x AVX-512 and 11.5x
  AVX2 speedups including encoding, with independent decoded-score/argmin and
  float-update checks. TQ8's previous AVX2 packed-scalar path improved about
  11.4–11.7x. These are inner-assignment results, not whole-index speedups.
- Query/corpus validation uses eight independent double sums at dimensions 128
  and above, preserving the original scalar calculation below that size and
  near decision thresholds. The standalone 32x128 validation measured 3.69 to
  0.85 microseconds; end-to-end comparisons must quantify its smaller contribution.
- The original YAML workflow is installed directly: named builds, serving
  variants, correlated rows, parameter grids, output layout, resume and plotting.
  `load_config` and `run` are the Python entry points. The three YAML examples
  ship with the wheel; no converter or source checkout is required.

Training pilot `arguana-training-pilot-001` completed ten objective/initialization
recipes. L2 and alternate IP seeding did not consistently improve the complete
build/recall tradeoff; keep the existing float/IP training defaults. Independent
float/TQ8/TQ4/TQ1 assignment is implemented with float centroid updates and norm/
weight semantics retained. The clean optimized phase
`arguana-quantization-packed-001` took 37.831/36.560/38.664/131.679 seconds, with
32-probe recall 0.946875/0.943750/0.938281/0.932031. The old float/TQ8 controls
finished, but an automatic apt update interrupted the TQ4 baseline. The failed
attempt is excluded. The clean TQ4/TQ1-only retry
`arguana-quantization-baseline-003` completed in 82.989/131.505 seconds. TQ4's
packed assignment gives a **2.146x whole-build speedup with an identical saved
tree hash and identical recall at every probe**; peak RSS is unchanged. TQ1 did
not improve in that snapshot. Subsequent TQ1 SIMD assignment in
`tq1-assignment-001` passed all eight original/decoded-codec oracles with exact
label equality and score tolerance 2e-6. Encoding plus assignment improved
2.36–3.51x for outer clouds and 2.88–4.62x for inner vectors across the two ISAs
and IP/L2. That candidate is integrated in `final-build-03`, with per-vector
norms applied before minima and unchanged float centroid updates. A fresh
TQ1-only whole-build/recall comparison completed in
`arguana-quantization-tq1-simd-001`: **131.505 to 31.005 seconds (4.24x)**,
with an identical serialized index hash and identical single/batch recall at all
four probe counts. Peak RSS is 1.275 GiB. This is an assignment optimization;
float remains the training default because its measured recall is higher.

Expanded serving microbench `serving-micro-001` stopped at a single/batch TQ1
disagreement. Narrow reproduction `leaf-tie-repro-001` proves full-k scores are
bit-identical on AVX2 and AVX-512; all top-10 membership differences are exact
cutoff ties. The TQ1 bucket evicts the first maximum-distance lane rather than
the largest tied ID, conflicting with the facade's score/ID order. The MVIVF
agent fixed tie-aware pruning/eviction/finalization while retaining grouped
scoring and removing its redundant/unmasked-tail max scan. TQ4/TQ8 heap cutoff
handling also honors nonmonotonic encoded IDs. Two clean standalone studies
(`bucket-001/002`) selected scalar reduction through k=32, fused AVX2 above that,
and two-pass AVX-512 above that. Default sorted callers use an explicit comparator;
compiled MVIVF requests an unsorted TQ1 leaf drain because its final global
selection supplies ordering. Native sorted/unsorted/sentinel and public tie
regressions pass in the combined checks above. Expanded `serving-micro-002`
completed 24 workers, 256 within-worker checks and 18 cross-ISA comparisons
without interference. All neighbor IDs match, within-worker scores are bit-exact,
and maximum cross-ISA error is 7.45e-8. This covers all four leaf codecs,
short/medium/long/ragged queries, root-M2M on/off and 1/4 workers; its mode-to-mode
timings are not original-versus-refactor acceptance evidence.

The final installed source archive audit (`sdist-final-01`) passes: 584 regular
files, required build/third-party/YAML inputs present, no generated datasets,
results, build directories or compiled artifacts. It is 2,198,613 bytes with
SHA256 `0d6f6a9fc906a71cdac01ae8c0b62cab11b5fe787443c5c2458736ec3d2b20d5`.

Serving investigation has identified the pilot regression rather than accepted
it. Fixed-tree cross-engine ratios were roughly 0.994–0.999; the independently
built new tree cost about 2.7% in either engine, with 13% more candidates visited.
Three root groups crossed the leaf-size cutoff, explaining most of the leaf-count
change. Reducing probes loses recall (`fine-probes-01`). Two initially unexplained
prediction differences were proven ties at the coarse rerank50 cutoff: visited
IDs and coarse scores match exactly, while original negative-score packed sorting
selects larger tied IDs and the new comparator selects smaller IDs. Details are
in `coarse-cutoff-01`; this is not a traversal or scoring discrepancy.

Final leaf-size studies `arguana-leaf100-final-01` and
`arguana-leaf95-final-01` completed eight fresh builds and forty searches over
all 1,406 queries with no busy guard samples. Both new builds at each setting
produce identical trees and predictions. Original trees vary across builds.
At leaf100, median build time is 38.173 seconds original versus 37.236 refactor;
median peak RSS is 1687.8 versus 1089.0 MiB (35.5% less). At probes16, median QPS
is 546.725 versus 540.162 with recall ranges 0.909673–0.910526 versus 0.909175.
The default setting remains roughly 1.2% slower at comparable recall; this is
not accepted as a speed win. Leaf95 lowers new build RSS by about 39% versus
its original controls. Compared with new leaf100 it serves 1.8–3.5% faster at
the same probe budget but loses 0.00320–0.00434 recall. Keep leaf100 default;
the leaf cutoff change does not provide a free improvement.

Remaining controlled measurements, sequentially in the benchmark agent's
exclusive native window:

1. Fixed-tree batch throughput from one to 96 physical cores, using
   `tests/refactor/BATCH_SCALING_PLAN.md`. The shared comparison driver records
   prepared NUMA/status sidecars outside timed loops. Then the bounded installed
   Python endpoint comparison, without redundant builds. The first 48-process
   block is running as `batch-scaling-final-01`; inspect dispersion before
   authorizing additional runs.
2. Isolated clean compilation/client rebuild comparison prepared in
   `/tmp/mvsic_compile_comparison.py`; it snapshots both source trees and leaves
   the original unchanged. Incremental `final-build-*` times are not compiler
   acceptance measurements.

All results belong below `/ssd2/kishen/mvsic_new/refactor`. Check the released EMD
marker and actual host load before timing. The user authorizes all physical cores
when free, and disjoint limited-core experiments when external sweeps return.
Keep shared-host results labeled separately. Ordinary correctness compilation
does not need a quiet timing window: `final-build-01` was unnecessarily stopped
by the strict guard during an OS update; `final-build-02` completed the build
without claiming a controlled compilation time. VNNI is compile-only on this
host; AMX remains a future capability placeholder.

## Continuation checkpoint (November 11, 2026)

Latest user decisions supersede the historical checkpoint below:

- Retrieval is IP/MaxSim only. Public MVIVF rejects L2, while shared distance
  primitives, MVClustering and inner k-means retain L2. Serving helpers are now
  concrete IP implementations, with no private metric branch or metric field.
- CPU/OS/backend selection is shared in `core/kernels/capabilities`, while
  method-specific factory wiring stays local. Only implemented backends can be
  selected. This host validates AVX2 and AVX-512; VNNI remains compile-only and
  AMX is a future capability placeholder.
- The installed benchmark must accept the **original YAML directly**. The user
  rejected replacing it with a new schema or requiring an import/conversion.
  Current Python work preserves named builds, serving variants, correlated
  search rows, Cartesian axes, output paths, CSV names, stop rules and plotting.
  Python orchestrates the workflow; shared C++ code times queries and batches.
  The public Python entry points are `mvsic.benchmark.load_config` and `run`.
  PyYAML is a base dependency; only plotting needs the optional plot extra.
- The active private trainer is `mvclustering/mvclustering_impl.h`. Abandoned
  lower bounds and the superseded regular legacy MVIVF implementation and its
  coupled diagnostics were removed. Flat/Spill IP code remains; generated L2
  retrieval translation units and public L2 aliases were retired.
- A fresh upstream fetch confirms Parallel-KMeans main is still
  `048e1ed6ff956febe80874c30f2178119ef104e4`. Training defaults remain unchanged
  until measured objective/initialization/assignment comparisons justify a
  change. Inner and outer assignment already support float/TQ8/TQ4/TQ1.

The first IP-only snapshot passed all five native suites on AVX2/AVX-512
(`refactor/ip-validation-01`), ten installed API/CLI tests
(`refactor/ip-wheel-01`), and all 84 strict original/refactor parity checks
(`refactor/ip-parity-01`). The incremental build took 113.50 seconds
(`refactor/ip-build-01`); it is not a clean compiler-cost comparison. Subsequent
concrete-IP cleanup, GoogleTest/concurrent-call coverage and original-YAML
integration still require the final rebuild and installed-wheel verification.
Lightweight tests currently pass (37 tests and 27 subtests); native YAML
end-to-end validation is being added.

The released EMD marker is still not an exclusive reservation. An unrelated
DynamicPiPNN sequence uses 64 workers on CPUs 0–63, overlapping every pilot CPU;
do not treat its low aggregate machine utilization as a quiet window. The
clustering agent has the first exclusive timing window for
`clustering_arguana_pilot.json`, phase training, output
`refactor/arguana-training-pilot-001`, using the unchanged installed ip-wheel-01
environment. Do not reinstall `/tmp/mvsic-refactor-env` while that experiment is
active. Other agents are restricted to source/docs/lightweight checks until it
finishes. The external supervisor's current final control run has a finite end;
check actual processes again before starting. No new performance claim is made.

The external pair finished and `arguana-training-pilot-001` has now started in a
quiet window. Keep that installed snapshot unchanged until its controller exits.
The benchmark package is source-frozen: 37 lightweight tests and 27 subtests
pass; both original-YAML and internal-schema end-to-end flows pass against a
temporary package overlay outside the checkout. This is integration evidence,
not the final PEP 517 wheel verification. The obsolete config converter and
public JSON example have been removed; three original-schema YAML examples ship.

A closer review of the interrupted serving pilot corrects the pooled own-build
ratio below: at exactly the same observed recall 0.90234375, original build03
reached 546.075 QPS and the two new records reached 535.879/527.482 QPS, a median
ratio of 0.97364 (about 2.64% lower). The older 0.9814 ratio also pooled the weaker
original build. Shared-tree serving at that probe was 1.0071. Root shape matches,
but new leaf vector counts have a larger p90. This is a hypothesis about extra
visited work, not an attributed cause. `arguana-native-01/serving-investigation-plan.md`
and `offline-structure-diagnostic.json` retain the analysis. A bounded cross-tree
ABBA comparison is being prepared before accepting any serving regression.

## Continuation checkpoint (November 10, 2026)

The source implementation has progressed beyond the original checkpoint below.
The EMDIVF marker has been released (`active: false`). The refactor-only GCC 12
build and all five native suites pass with both AVX2 and AVX-512 selected; logs
are under `/ssd2/kishen/mvsic_new/refactor/gcc-validation-01`. Twenty-eight
lightweight benchmark/provenance tests also pass in `/tmp/mvsic-refactor-env`.
The Clang build also passes all five suites with AVX2 and AVX-512 selected.
The installed wheel passes ten API/CLI tests outside the checkout, an installed
C++ consumer configures/builds/runs, and strict original/refactor parity passes
168 checks. **Performance acceptance and training experiments remain pending.**
The primary comparison tree is now `build/refactor-clang`, using
Clang 15 for both implementations: GCC rejects the original header's in-class
explicit specializations. The original checkout remains unchanged.

- Method folders now live under `mvsic/algorithms/`. New MVIVF is in
  `algorithms/mvivf/`; its clustering facade and private trainer are in
  `algorithms/mvivf/mvclustering/`. Old MVIVF variants/wrappers are in
  `algorithms/mvivf/legacy/` pending validated replacement and cleanup.
  The clone's EMDIVF sources have also moved to `algorithms/emdivf/`; the original
  experiment repository and its coordination marker remain untouched.
- Build, search, v4 persistence, grouped batch traversal/root M2M, quantized
  serving, and independent assignment codecs are implemented as source.
- CMake/PEP 517 packaging and the single Python binding are wired. Actual serving
  and clustering ISA variants use isolated DSOs and a shared Parlay scheduler
  overlay. Small Intel dot functions share one target-attributed source file.
- The common native benchmark loop is `algorithms/benchmark.cc`; installed Python
  commands remain in `benchmark/`. Single latency, parallel latency, and batch use
  the same implementation with explicit scheduling. Runtime fingerprints include
  shared libraries, and comparison runs pin their DSOs as well as executables.
- K-means helpers and the required upstream source subset are in `core/kmeans/`;
  the external Parallel-KMeans build dependency has been removed. The user has
  identified its paper as UltraKMeans (PPoPP 2027, to appear) and requested an audit
  of updated upstream code plus controlled L2/IP/initialization build-quality
  experiments. This is additional authorized work, with existing defaults held
  until measured evidence supports a change.
- Updated upstream UltraKMeans was audited in an isolated checkout at
  `/tmp/mvsic-parallel-kmeans-latest`, revision
  `048e1ed6ff956febe80874c30f2178119ef104e4`. Experimental `inner_metric`,
  `inner_init`, and `inner_backend` controls preserve existing defaults. Safe
  initializer assignment reuse, adaptive float blocks/means, shared immutable
  encodings, and direct row-mapped serving encodes are source-level candidates;
  no performance benefit has been measured yet. The copied baseline subset stays
  pinned for original/refactor comparisons.
- Inner Lloyd training skips the final assignment when its labels are unused
  and verbose costs are disabled. Raw-center search skips quantized-root scratch.
  Container review preserves construction for owners and zeroing for accumulators
  and codec padding; only fully overwritten primitive scratch can be uninitialized.
  Standard-vector gathers now construct floats from input rows without first
  zeroing the complete output. The GCC native regression checks pass.
- Original/refactor comparisons now include both shared-original-tree serving
  and each implementation's independently built trees, with separate aggregation
  and recall matching. The latter is required to assess construction quality.
- Native correctness and original/refactor tooling are under `tests/refactor/`,
  with Python ownership/installed CLI tests under `tests/`. Installed checks run
  in the separate `/tmp/mvsic-refactor-env` conda environment. `build/refactor`
  is the GCC diagnostic tree.
- The queued build completed with initial integration errors. Corrected Parlay
  slice pointer access, the TQ8 header's missing primitive declarations, and the
  shared scheduler's global-dynamic TLS model (required by PIE and Python dlopen).
  Initial/retry logs remain under `/ssd2/kishen/mvsic_new/refactor/compile-*`.
- The guarded eight-job Clang build completed. Logs are in
  `/ssd2/kishen/mvsic_new/refactor/clang-build-01`; its combined dependency,
  original/refactor and test build took 288.36 seconds. This is diagnostic
  evidence, not an isolated compilation comparison.
- Validation found an inherited TQ4 AVX-512 IP padding bug: zero padded lanes
  could beat every real negative dot product. Padding now has a negative norm
  sentinel and the scoring epilogue masks it using already loaded metadata.
  TQ8 already handled this correctly. Independent scalar panel oracles cover
  antipodal, zero and mixed inputs with partial/full panels and grouped queries.
  The corrected native checks are in `refactor/clang-tq4-validation`; the
  rebuilt installed wheel checks are in `refactor/wheel-validation-02`.
- Strict parity is in `refactor/parity-03`: 168 checks, maximum score difference
  4.77e-7, matching partial-search visited sets. Its correctness reference is
  the untouched original AVX2 single-query path. Original AVX-512 has the TQ4
  padding bug and original AVX2 TQ1 batch returns placeholder maximum scores.
  The explicit `--reference-single-only` flag excludes only that unsupported
  original batch self-check; every new batch mode remains checked. Performance
  comparisons must still use the original native AVX-512 binary.
- `refactor/cpp-install-01` verifies the exported CMake package from an external
  consumer without source-tree dependency includes. The current CMake cache
  was last configured by wheel building (testing/comparison OFF); re-enable
  those options before rebuilding native test or comparison targets.
- The released marker is not an exclusive reservation: subsequent original
  EMDIVF and unrelated `stream-baseline` jobs were observed. Check host processes
  as well as the marker and obtain a quiet window before timing comparisons.
- A quiet-window ArguAna native pilot started at about 03:17 UTC November 11,
  under `/ssd2/kishen/mvsic_new/refactor/arguana-native-01`. Four builds and 49
  search processes completed before the controller was interrupted. A new
  unrelated DynamicPiPNN `stream-baseline` candidate-r4 job started around
  03:22:15 UTC using 64 workers on CPUs 0–63, overlapping our affinity. The
  controller and its active child were stopped cleanly; `contamination.json`
  marks the run unsuitable for acceptance. Do not resume or publish that run
  as a complete controlled comparison. No heavy refactor process remains.
- The uncontaminated build pairs took 39.261/39.429 seconds for the original
  and 38.649/38.606 for the refactor. Peak process RSS was about 1.76/1.79 GB
  versus 1.16/1.17 GB. Clean shared-tree, single-thread points at probes 4/8/16
  had identical recall and QPS ratios 1.0009/1.0020/1.0071. These are partial
  diagnostic observations, not performance acceptance. Exclude the later
  probes-32 and eight-thread measurements affected by the competing job.
  `diagnostic-summary.md/json` retain the conservative first 24 search records.
  At own-build probes 16 the refactor reached the stronger original recall
  (0.90234375), with a preliminary QPS ratio of 0.9814; repeat this point in the
  next quiet window rather than declaring the small difference acceptable.
- Original builds differed (433 versus 454 nodes); new builds were byte-identical
  (434 nodes). Source inspection found unsynchronized process-global empty-center
  seed counters in original `core/utils/lloyds_kmeans.h`, reached concurrently
  by outer clustering. The new counters are trainer-owned. This explains a
  plausible source of the original variation but has not been instrumented to
  attribute each changed center; independently built-tree recall remains required.
  Another original defect leaves trailing fixed-width centroid slots uninitialized
  when a replacement document is short. New code fills them by repeating real
  vectors, which preserves directed-min scoring. This case can occur with
  ArguAna's centroid width 140; its actual contribution to these builds has not
  been instrumented. Other default grouping, weighting, normalization, seeding
  and ordinary tree-limit behavior match for this corpus.
- Next quiet-window work: rerun `tests/refactor/configs/arguana_native_pilot.json`
  into a new directory, investigate regressions, then run the installed-Python
  follow-up and the ten-recipe `clustering_arguana_pilot.json` training study.
  The latter fixes IP serving and changes inner objective/initialization/backend;
  select a recipe explicitly before its fresh float/TQ8/TQ4/TQ1 assignment phase.
  Defaults remain unchanged. An isolated eight-job compiler measurement script
  is prepared at `/tmp/mvsic_compile_comparison.py`; it copies sources into `/tmp`,
  warms native dependencies separately, records clean library/client and
  public-header/implementation rebuild costs, and leaves the original untouched.
  Invoke it with a new output directory below `/ssd2/kishen/mvsic_new/refactor`.
  Its original workload is the IP production subset, not the full legacy
  wheel. All heavy phases must run sequentially, without competing experiments.
- All implementation, cleanup, documentation and static Bazel integration changes
  are staged on `reformat`; no commit or push. The last lightweight run passed
  28 tests and 22 subtests. The CMake native binaries and installed wheel match
  the tested implementation; later edits only affected docs/optional legacy
  wrappers/Bazel rules, including retaining assertions in optimized test builds.

## Folder completion and cleanup

The user authorizes removing superseded legacy files once their replacements
are verified and tested. Complete that cleanup autonomously, stage the changes,
and notify the user as each folder is finalized. `python/` is finalized for the
current MVIVF API and the user has been notified. Runtime acceptance is pending.

| Folder | Remaining completion checks |
| --- | --- |
| `core/kernels/` | Native dispatch/panel/shared-scheduler tests pass on AVX2 and AVX-512; performance review remains. VNNI/AMX are not hardware validated. |
| `core/kmeans/` | Native assignment, weighting, norm, initialization and buffer checks pass; controlled build-quality experiments remain before changing defaults. |
| `algorithms/mvivf/mvclustering/` | Native clustering checks pass; build/recall comparisons and required private-trainer naming review remain. |
| `algorithms/mvivf/` | Correctness and original parity pass; generated regular-wrapper cleanup is complete. Speed/recall/memory acceptance remains. |
| `python/` | Finalized: ten installed API/CLI checks pass; 25 superseded registrations and `bind_macros.h` removed. |
| `benchmark/` | Installed config build/search/resume/plot and parser/provenance checks pass; real-data native timing comparison remains. |

Deletion must follow dependencies, not filenames: `legacy_mvclustering.h` still
implements the compiled trainer, and `core/legacy_index.h` still supports other
algorithm families. Preserve required implementations while removing redundant
wrappers and registration slabs. Folder moves alone do not establish completion
for the other algorithm families. The original comparison repository is read-only.

Cleanup removed 72 regular-MVIVF generated TUs (`bench_inst/*_regular_*.cc`:
36; each search instantiation folder: 18), two obsolete standalone wrappers,
and their matching generator, declaration, dispatch and BUILD entries.
Shared files still serve Flat/Spill; their 144 remaining TUs regenerate
byte-identically. Regular legacy tree headers still serve diagnostic
microbenchmarks and the cloned EMDIVF bucket pilot, which access internal tree
state. They remain optional until those consumers are ported or retired.
All 25 cloned Python `register_*.cc` files and `bind_macros.h` were removed after
installed API acceptance. The baseline comparison
uses the original repository's registration files. Record retirement of old
regular-MVIVF codecs beyond float/TQ4/TQ1/TQ8 explicitly rather than claiming
replacement coverage for them. The inventory also caught and corrected the
new `core/index.cc` accidentally including `legacy_index.h` instead of `index.h`.

Use the current README and source paths for the revised layout. The following
sections preserve the earlier handoff and explain the original partial work.

The user requested this separate clone for the refactor. This is an **uncompiled,
incomplete implementation checkpoint**, not a working or performance-validated replacement.
No commits or pushes have been made. Review the staged diff before continuing.

## Locations and coordination

- Work only in `/home/kishen/MVSIC-new/MVSIC`, branch `reformat`.
- Original implementation and comparison baseline: `/home/kishen/MVSIC`.
- `data -> /ssd2/kishen/multivector` shares the original data store.
- `results -> /ssd2/kishen/mvsic_new` isolates all new results.
- Original architecture findings: `/home/kishen/MVSIC/docs/architecture_review.md`,
  `architecture_core_audit.md`, and `architecture_python_audit.md`.
- EMDIVF experiments continue in the original repo. Before heavy compilation or
  any performance experiment, read
  `/home/kishen/MVSIC/emdivf_experiments/artifacts/EMDIVF_BENCHMARK_ACTIVE.json`.
  While `active` is true, restrict work to editing and lightweight inspection.
  Do not disturb those latency measurements. The original chat updates the marker.

## User priorities

Runtime speed first; simplicity/readability second. The replacement must match or
beat the original MVIVF at comparable recall before it is accepted. Preserve
existing class/function/variable/file names and commenting style where practical.
Validate ownership, shapes, configurations and persisted inputs once at boundaries;
use established invariants inside hot loops rather than adding repeated checks.
Do not replace SIMD, blocked scoring or batch traversal with scalar convenience code.
Undefined behavior should be removed by using the correct extents/representation,
not by adding checks to every element operation.

Start with MVIVF, Python usability and an installed config-based benchmark interface.
Prefer `mvsic/{core,algorithms,benchmark,python}` with few directory levels. GPU is a
future backend, not part of this first implementation. Keep private hot-kernel metric
specializations if necessary; avoid the metric × center codec × leaf codec × family
Cartesian product throughout the algorithm, bindings and benchmark wrappers.

## Files started

- `mvsic/core/{types.h,data.h,data.cc,index.h,index.cc}`: runtime `Metric`,
  `ClusteringQuantizer`, metric-neutral point-cloud views, owner retention, common
  non-template `Index`, and RAII PCS loading. Public headers use the standard library.
- `mvsic/algorithms/mvivf.h`: non-template `IndexMVIVF` with opaque implementation.
  **Its compiled build/search/kernel implementation is still pending.**
- The old base is retained in `mvsic/core/legacy_index.h` as `LegacyIndex`; old tree
  consumers and explicit instantiations were mechanically renamed to
  `LegacyIndexMVIVF` so they can coexist with the new public names. These edits are
  compatibility scaffolding, not an intended final collection of generated files.
- `IndexParams::mvivf` keeps its signature and now defaults to OneBitTQ leaves;
  `quantize_centers` stays true. `SearchParams` scalar initialization was corrected.
- `mvsic/python/bindings.cpp`: one new, uncompiled binding TU for the non-template
  API, NumPy inputs/results, owner retention and GIL release around native work.
  It requires `MVClusteringConfig.assignment_quantizer`. It has **not** been wired
  into the old Python BUILD or tested. `mvsic/__init__.py` now lazily imports
  `mvsic._mvsic`; the new extension must be installed there.
- `mvsic/benchmark/{config.py,cli.py,__init__.py,__main__.py}` plus `mvsic/cli.py`:
  versioned JSON/YAML config parser and CLI skeleton. Only validation is implemented;
  `runner.py`, build/search/plot execution, examples and tests are pending.
- `mvsic/core/kernels/`: initial Intel capability detection and compiled scalar,
  AVX2, AVX-512 and VNNI dot-product dispatch sources/tests. See
  `REFORMAT_CLUSTERING_HANDOFF.md` for exact status; **none compiled or tested**.
  AMX is explicitly unavailable, not a claimed implementation.
- `MODULE.bazel` and `third_party/rabitq/`: copy of the dependency compatibility pin
  and seeded-rotation patch already validated for EMDIVF in the original repo.
  This new clone itself has not been built. RaBitQ commit:
  `abd9ddb61c9267b547e7486a3f526336fd1177e2`.
- `REFORMAT_CORE_HANDOFF.md` and `REFORMAT_CLUSTERING_HANDOFF.md` contain the agents'
  final implementation notes if present. They take precedence over inferred status.

## Settled C++ interface

`IndexMVIVF(uint32_t d, Metric, const IndexParams& = IndexParams::mvivf())`.
`build(PointCloudSetView)`, `load(path, PointCloudSetView)`, `save(path)`.
`search(PointCloud, points, SearchParams)` returns `vector<pair<uint32_t,float>>`;
there is also a bound-corpus overload. `search_all` has corresponding batch overloads.
`search_with_stats` returns `(neighbors, size_t, vector<double>)`, keeping old timers.

`PointCloud` fields: `values,n,d,id`.
`PointCloudSetView` fields: `n,d,values,offsets,ids,owner`.
Offsets count **float coordinates**, matching PCS, not vector rows.
An index must retain the bound view owner; queries borrow storage for the call.

The current binding returns `(ids, distances)` NumPy arrays, batched arrays for
`search_all`. It exposes the existing `IndexParams.mvivf` and `SearchParams.mvivf`
factory names. The native constructor reads `params.pq.method` and
`params.quantize_centers`; the binding exposes `params.quantizer` as a property.

## Next work

1. Inspect the entire staged diff and agent handoffs; fix scaffolding rather than
   assuming it works. Finish compiled MVIVF topology/build/search with coarse codec
   operations and the optimized grouped-leaf / root many-to-many paths retained.
2. Complete a compiled MVClustering facade with independent outer/inner assignment
   choices float/TQ8/TQ4/TQ1, preserving actual float centroid updates, seeding,
   weighting and normalization semantics. TQ1's unit-norm assumptions require care:
   centroid vectors are generally not unit norm. Do not silently normalize them.
3. Wire a single native binding and installable Python package, preferably a proper
   PEP 517 native wheel with accurate ABI/platform tags. CMake/scikit-build-core and
   new Python BUILD changes have **not** been implemented. Keep dependencies lean;
   make embedding tools, external baselines, plotting and YAML extras optional.
4. Finish installed `mvsic benchmark validate/build/search/plot --config PATH`;
   strict schemas, resolved paths, input/build fingerprints, full-precision raw
   results, failures and safe resume. The starter parser intentionally uses one
   small version-1 schema; compatibility with old experiment YAML is still pending.
5. Keep Intel ISA kernels compiled separately with selection outside hot loops.
   Distinguish CPU support, OS state, compiled implementations and requested backend.
   AMX also needs Linux XSTATE permission and suitable batch/panel shapes; CPUID alone
   is insufficient. This host has AVX-512 but lacks VNNI/AMX. Design these paths, but
   do not claim hardware validation that has not happened.
6. Once the EMDIVF coordination marker is false, build and run meaningful correctness
   tests: PCS/mmap and NumPy ownership/strides, IP/L2, raw/quantized paths, save/load,
   query/batch consistency, exact small-data oracle and original-MVIVF parity.
7. Compare clean/incremental compilation, build time, native and Python latency,
   batch throughput, memory and recall/work curves against the original folder on
   identical data/settings/CPU placement. Fix regressions before declaring success.
8. Remove obsolete generated MVIVF wrapper/registration files after the new route
   works. Stage the required final changes; **do not commit or push**.
