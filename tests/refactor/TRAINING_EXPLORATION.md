# Training exploration — November 12, 2026

These are bounded research screens, separate from serving-refactor acceptance.
Training defaults remain unchanged. The separately validated IP-only API cleanup
retires public Metric/L2 selection; these experiments use frozen earlier runtime
snapshots. Results live under
`/ssd2/kishen/mvsic_new/refactor`. Runs use unpinned workers on the shared host;
timings are exploratory, not clean-host performance claims.

Current direction: retain automatic representative sizing (`s=0`). The user
closed fixed-size tuning on November 12; the size/ratio/per-level studies below
are historical evidence, not pending work or recommended configurations. The
shared-runtime fix is validated. Current work tests exact copy/encoding reuse
and outer quantized assignments with float refinement and float inner training.
No experimental production default has changed.

## Resumed exact-work screens

`final-root-float16-screen-01` passes one new bounded C5 screen. The production
Float e174 root retains all241 requested centers, so its published pool is the
complete final-assignment state. Production Float replays every saved label on
4,096 unique documents. TQ1 top16 followed by exact Float refinement recovers
4,094 labels (99.951%); reported Float scores match the direct-pair oracle.
The two misses have regrets.005405/.034889, making mean regret9.837e-6 only
just below the declared1e-5 gate. This is not retrieval-quality equivalence.

Median assignment cost is.73637 s Float versus.32709 s including TQ1 center
encoding/panels/coarse scoring/refinement (2.251x). Adding immutable model/query
encoding gives.46328 s (1.589x). Output-workspace allocation, input gathering
and teardown are excluded explicitly. Every sample remains retained. Only one
fixed shortlist size and one real-data case ran; AVX512 only, not VNNI/AMX.
The single private whole FiQA build changes final ROOT assignment only: all
training and all nonroot assignments remain Float. It completes in44.771 s,
including root-capture I/O, with7627.10 CPU-s and11,154,018,304B peak RSS.
This unmatched measurement establishes no build speedup. Actual C5 centers
are byte-identical to Float;37 of57,638 final root labels change. All648-query
p16/p32 hits are3037/3729 versus Float2991/3711. Every shared returned ID has
exact score bits, and all serial repetitions reproduce prior prediction bytes.

The balanced eight-sweep serial comparison at the same p32 fails the declared
gate. Float→candidate mean is3.378559→3.410189 ms, p95
4.565156→4.634705 ms, and p99 5.193126→5.308708 ms; median full-sweep time
is2.184009→2.210827 s. All four local pairs are slower. The quality increase
does not excuse these regressions: no batch, alternate probe, retry or default
promotion. Evidence: `root-final-float16-build-01`. The next bounded screen uses
TQ8 top8 plus Float refinement and mapped previous centers on the genuine C1
state, keeping inner training Float. No experimental defaults have changed.

`outer-tq8-float8-c1-screen-01` completes that single screen. All4096 genuine
C1 Float positional winners and score bits match, with zero regret and8.011
mean exact Float calls. Fifteen norm/zero/tie/ragged-tail fixtures pass; there
is no duplicate inner-vector query cache. However, cached median assignment
takes1.035914 s versus Float.714573 s (45% slower). Coarse scoring/query packing
takes.984–1.031 s, while Float refinement takes about.030 s. One-time model and
query encoding add.101613 s. All three samples are retained. The declared cost
gate fails: no whole build, retry or codec/shortlist grid. Next is a source-only
assessment of exact Float query batching to amortize repeated center packing.

Distance-reuse source audit: inner Lloyds already skips its unused final
assignment when quiet, and computes costs only when requested by verbosity.
Its scored centers precede the final float mean/normalization, so those distances
cannot safely seed the next outer assignment. Sampling and document weights
also differ. Root Float top1 instead repeats center-matrix packing per document;
the existing blocked AllPairs path provides a bounded control for a streaming
query-batching pilot. Grouping changes Eigen's kernel/tail choices, so exact
formulas alone do not prove score-bit or whole-tree identity.

`exact-batched-float-c1-screen-01` compiles in21.35 s and stops at its12-case
fixture gate, before real C1 timing. Labels all match. The grouped AllPairs
control and streaming candidate each have two winner-score bit differences;
only the first witness is recorded: d7, one center, row10, negative zero versus
positive zero. The report does not classify the second difference. This is no
speed or general correctness result. A narrow follow-up will classify every
mismatch and retain the old kernel for single-center shapes, where grouping
is unnecessary for the proposed large-center-count optimization. No production
source has changed.

`exact-batched-float-c1-screen-02` completes that narrow follow-up. The unchanged
AllPairs control classifies both fixture differences as signed zero: no labels
or nonzero score bits differ. Retaining production Top for a single center gives
the streaming candidate exact bits in all12 fixtures. All4096 actual C1 winner
labels and score bits also match production Top and the saved oracle. However,
its512-vector grouping takes.831355 s versus Float.713752 s,16.48% slower;
all three local pairs regress12.96%/12.40%/19.69%. Existing AllPairs+argmin
takes.698255 s versus its paired Float.728621 s, only1.0435x, below the1.10x
gate. All samples remain archived. No block-size grid, whole build or production
change follows; close this branch and retain the validated Float default.

Two independent fresh FiQA pairs use the corrected installed runtime, 192 Parlay
workers and one ancillary worker. Each compares the current implementation with
one private change; their gains must not be multiplied or called a combined
installed result.

| Change | Build wall before / after, s | Build CPU before / after, s | Peak RSS change | Saved tree |
|---|---:|---:|---:|---|
| Borrow const row views before inner sampling |49.101 /45.761|8806.721 /8168.017|−23.29%|Float e174fc03, exact|
| Reuse root TQ1 codes in inner kmeans |28.107 /25.867|4761.502 /4388.192|−1.49%|TQ1 dad38138, exact|

Views retain Parlay scan/fill order, duplicates and weights; only selected rows
are copied into the unchanged float backend. The cache consumes vector-index
views immediately and retains pointers to build-lifetime root codes. Its counters
show one root encoding of 7,314,136 vectors, 14,020 inner cache batches serving
21,984,251 vector occurrences, and zero new inner query encodings. Changing
centers are still encoded and rescored. Focused ownership/order oracles and the
existing19 clustering tests pass for both candidates. The combined five-file
implementation and two existing-suite regressions are applied. Full selected
Clang/GCC/host-ISA/package validation passes, with both combined tree hashes
exact. Source 65ed65ad and the normal installed wheel exclude unaccepted Flat.
Final installed serving acceptance remains before staging.
Evidence: `borrowed-rows-queryword-01` and `tq1-inner-cache-01/fiqa-pair-01`.

The initial integrated source65ed loses3.35% median batch QPS versus original
and1.63% in direct previous-runtime isolation. Those observations remain retained.
Panel alignment yields only1.7–7.4% micro gains and stops at its declared threshold;
function alignment64 loses5–7% serial latency and is rejected. The main-model
p30 quality point misses the original p32 quality gate by8 of14,060 hits; no
alternate probe is tried. No rejected variant is applied.

A current-core phase probe measures serial request construction at1.924 ms,
more than the entire query-preparation phase1.716 ms. Flat Parlay grouping now
replaces serial per-leaf vector growth in one existing.cc file, preserving all
score/scatter inputs. Its three-case serving gate passes: median batch QPS
15,010 versus original15,005 and source65ed14,283. Serial mean/tails remain
3.92%/2.74%/2.56% faster than original, with higher recall and exact current
prediction bytes. Mean batch QPS is still1.86% below original; serial mean is
1.93% slower than65ed. These are bounded shared-host observations, with every
earlier adverse sample retained. Source78d7 passes final GCC/package validation;
the six verified source files and reports are staged. Evidence:
`parlay-requests-final-validation-01`, `mvivf-batch192-current-phase-01`, `parlay-requests-01`
and `selected-view-cache-parlay-grouping-serving192-01`.

The fresh frozen-original FiQA Float build completes in 57.064 s. Through the
same current serving runtime, all 648 queries give original/current Float recall
45.617284%/46.157407% at p16 and 56.913580%/57.268519% at p32. Current/View trees
are identical. All 3,450/4,156 shared returned IDs at p16/p32 have bit-identical
reranked scores; different trained trees change candidate discovery. The original
has 12,644 zero center vectors and 6,198 nonzero vectors outside unit norm tolerance;
current has neither. Both cover all 57,638 documents exactly once. These facts
do not causally apportion every neighbor difference. Predictions, audits and
offline score comparison are retained in `borrowed-rows-queryword-01`.

The quality-first private outer policy has passed 24 focused checks and one
whole FiQA build in 40.925 s. It uses exact Float initial/final assignments,
TQ1 top8 plus Float refinement and the mapped genuine previous center on four
intermediate passes, and unchanged weighted Float legacy10 inner training.
Full 648-query recall is 46.589506%/57.978395% at p16/p32, versus Float
46.157407%/57.268519% and coupled TQ1 42.561728%/53.456790%. This recovers Float
quality in this screen. Same-p32 serial pooled mean/p95/p99 improve, but median
full-sweep latency is 1.20% slower; the first Float sweep is an outlier despite
full warmups. Do not claim a clean latency win. One justified p31 quality point
reaches 3718 hits versus Float p32's 3711. The balanced matched-recall comparison
improves median full-sweep time 1.997134→1.962943 s, mean 3.086404→3.037021 ms
and p95 4.186146→4.150279 ms, but p99 worsens 4.714289→4.852458 ms (+2.93%).
All eight sweeps reproduce their models' prior ID/score bytes. The declared
strict gate fails: no batch, repeat, alternate probe or default promotion.
No probe grid runs.
A Python RTLD_LOCAL ownership-audit preflight
failed before data or queries; failed files are retained. A narrow existing-core
RTLD_NOLOAD handle check passes, preserving mapped libraries, and the actual
quality attempt completes. Source and evidence:
`outer-refinement-float-inner-build-01`.

The runtime wake-up defect is confirmed by `waiter-runtime-build-01`: the same
compiled backend object, four-worker scheduler and32 finite jobs use one worker
before a narrow waiter-table relink and all four afterward. ISA libraries had
private libstdc++ wait tables despite shared Parlay TLS. The borrowed-row preload
also interposes that table, invalidating copy-only speed attribution. Preserve
its exact trees and memory observations, but remeasure timings after the runtime
is corrected. The phase-trace preload has its own local waiter table, so it too
cannot identify the cause of the uninstrumented serving transient.

`waiter-runtime-arguana-01` is the no-preload real-build gate. Float and TQ1
trees match their exact anchors. One before/after pair measures Float6.528/6.402 s
and TQ1 5.319/5.282 s, with similar CPU and peak memory. It demonstrates exactness
and normal operation, not a large speed gain or a performance median; the long
diagnostic stall is absent from both controls.

The large TQ4/TQ1 speedups below compare optimized kernels with the earlier
implementation of the SAME codec. The original eight-worker float build was
37.831 s, compared with 38.664 s for optimized TQ4 and 31.005 s for optimized
TQ1. Thus the float-relative gain is about 18% shorter build time for TQ1, with
lower recall, and no build win for TQ4. The often-quoted 16.018 versus 23.416 s
refinement comparison spans different screens/initialization controls; it is
not a clean paired speedup. No large equal-recall build win is established.

Timing correction: older `ClusteringIterationStats.centroid_update_seconds`
includes assignment, because the accumulating Parlay timer was restarted
without a reset. For those archived per-iteration diagnostics, subtract
`assignment_seconds` to recover exclusive update time. Build wall/CPU, tree
bytes and recall measurements are unaffected. The FiQA TQ1 root therefore
spends about 11.67 s assigning and 2.61 s updating, not 26 s total. A separate
source fix passes19 clustering tests on both host ISAs, and the old trainer
fails the focused regression (`clustering-timer-validation-01`). Profiling wrappers must
label the old inclusive field explicitly.

## Decisions and evidence so far

- The corrected runtime passes the192-worker default build and fresh-own-model
  p32 serving gates; this checkpoints the refactor, not a quantized default.
- `runtime-cluster-profile-01` completes one FiQA TQ1 AUTO build in 35.618 s,
  including the diagnostic proxy. No quality has yet been evaluated. Root
  clustering takes 20.248 s, with measured five-iteration assignment/update sums
  of 11.772/2.607 s. Assignment is 81.9% of those disjoint iteration phases, not
  of total build time. The 5.866 s residual includes final full-data assignment
  and other untimed work; it is not an isolated assignment measurement.
  Factory preparation/release take 1.013/.523 s. The 217 nonroot calls span 156
  workers with 141 concurrent calls on one 192-worker pool; their 12.848 s envelope
  must not be replaced by the sum of overlapping node durations. Peak through
  build is 14.144 GiB.
- `runtime-cluster-profile-float-01` completes the matching Float reference and
  common-current-runtime quality screen on all648 FiQA queries. Float/TQ1 build
  times are45.880/35.618 s with the same diagnostic proxy; peak memory is
  12.383/14.144 GiB. Float/TQ1 recall is.461574/.425617 at p16 and
  .572685/.534568 at p32. Coupled outer/inner TQ1 is22.4% faster in this one pair
  but loses3.6/3.8 recall percentage points and uses more memory. Do not promote
  it. Float root measured assignment/update sums are18.652/2.220 s; assignment
  is89.4% of those phases. Its29.905 s root call has9.031 s outside the measured
  phases, including final assignment and other work. These observations do not
  isolate outer versus inner quantization's effects on quality or update cost.
- Outer root ANN remains the first large work-reduction experiment. The prepared
  pilot uses stock directed Vamana mechanics with a Float representative cache,
  norm-aware TQ1 traversal and identical Float4 refinement versus its dense
  control. The first real screen will reuse published Float root centers and
  exact root-child partition labels from that saved model, avoiding another
  training/capture compile. Its241 clouds each contain127 vectors. Start with
  4,096 unique sampled query documents; retain every representative unchanged.
  Dense Float must reproduce extracted labels before approximate scoring.
  Encoding/cache/graph/refinement costs are counted, and duplicate/
  quantization-collision fixtures must pass before a real snapshot is scored.
  The first screen is recorded below; no whole-tree speed or quality gain is claimed.

The first outer ANN screen has now run (`outer-assignment-ann-gate1-01` and
`outer-assignment-ann-published-root-4096-01`). Native norm/tie/duplicate and
quantization-collision fixtures pass. Production dense Float reproduces all
4,096 sampled saved assignments exactly. The stock directed R8/L16/alpha1
graph stops before query traversal at the connectivity/cache guard. A bounded
same-binary debugger diagnostic resolves the cause: cache misses 0, hits 26,342,
but entry 0 reaches 196 of 241 representatives. No ANN query/quality/throughput
result exists; the failure does not reject outer ANN generally. Do not expand
to all 57,638 queries or whole-tree training on this configuration. Both the
original failed process and diagnostic are preserved. Dense Float/TQ1 controls
completed, but their independent quality verifier was not run after this stop.

The user prioritizes previous-assignment starts next, transferring the inner
kmeans approach. Use real C0 assignments with updated C1 centers, starting from
both the shared entry and the previous center. A necessary ownership/detail:
MVClustering compacts occupied groups into ascending label order during updates,
so previous center positions must be remapped to that group order. Do not seed
with saved current-state exact labels. Warm starts improve candidate discovery
but cannot guarantee discovery of a newly best unreachable center, and initial/
unsampled final-pass points have no previous assignment. These cases and the
prepared add-only connectivity repair need explicit separate treatment.

Existing raw control outputs have3125/4096 exact positional label matches for
TQ1 alone and3949/4096 with Float4 refinement (76.29% versus96.41%). This is
agreement against the replayed saved Float assignment, not retrieval recall or
an independently verified tie-aware all-pairs result. It motivates testing float
refinement and preserving float inner updates before more graph tuning.

`outer-assignment-warm-c1-4096-01` completes the genuine previous-center test.
Capture uses the unchanged full24,100-row Float training sample; the4096 chosen
training positions contain3953 distinct documents. C0 seed clouds match corpus
bytes, previous labels precede the C1 update, and the exact compaction map is
validated. Two empty clusters move32 occupied center positions. One graph/state
serves paired entry0 and previous-plus-entry0 batches with identical Float4.

Warm starts raise independent-oracle assignment agreement3339→3916 of4096
(81.52%→95.61%); dense TQ1+Float4 recovers3997 (97.58%). Mean regret remains
.000490 versus dense .000154, so the quality gate fails. The previous-start
union reaches the exact winner for4049 queries versus3794 with entry0 only;
47 warm misses are unreachable, another133 are reachable but lost to search/
shortlist selection. Returned Float scores match the independent oracle bitwise.

Warm mean scored centers is56.41 versus62.17 cold and241 dense. Its query work
is3.32x faster, but the complete graph/cache rebuild costs.13806 s; cached state
cost is.25230 versus.27357 s dense, only8.4% faster. Canonical setup in the dense
experimental control is avoidable in a production version, making that small
ANN win optimistic. These subset costs do not establish a universal24k-query
penalty. The predeclared quality/cost stop stands; no whole training or retrieval
gain is claimed.

Offline reuse of saved all-pairs scores predicts that Float-refining top4 plus
the genuine previous center would recover3951 warm/4040 dense assignments
(96.46%/98.63%), with.054/.065 extra Float evaluations per query on average.
This recovers35/43 winners but leaves a gap; it is a counterfactual, not another
native experiment or timing result. No parameter grid or connectivity repair ran.

The bounded exact-kernel test changes only the private TQ1 query-word copy
to a constant four bytes. Static codegen shows runtime-sized `memcpy` calls in
both ISA tile loops; private Model-produced queries already have64-byte zero
padding. Keep the database-tail copy bounded and preserve all float arithmetic,
masks and ties. Source-level padding proof and an extracted-kernel microbenchmark
were prepared under `/tmp/mvsic_tq1_queryword_prep`. Those gates now pass:
actual-Model release score/label checks, both literal-kernel sanitizer and guard-
page tests, MC19 against the actual candidate backend, and exact FiQA model bytes.
One no-probe TQ1 pair at192 workers falls35.494→24.352 s, CPU6310→4194 s, with
peak+0.085%. This is an exact within-TQ1 gain, not a quality/default change.
`tq1-queryword-validation-01` passes the full compiler/host-ISA native matrix,
twelve prediction-byte cases, installed API/YAML and external consumers. The
patch is staged; current environment `/tmp/mvsic-queryword-env`. The old full-
Model ASan compile cap remains incomplete, separately from the passing narrow
sanitizer/protected-page composition.

Encoding is already FHT-based. One-time FiQA TQ1 root preparation measured1.013 s;
repeated inner and center encoding have not been isolated. Inner query vectors
encode once per new inner call and reuse codes across its Lloyd steps. Source
audit proves immutable root TQ1 codes can be reused with TQ8's existing vector
mapping while preserving original encoder seed/norm semantics. Actual trainer
seed increments during initial/empty-cluster reseeding must not be used as codec
seeds. Changing centers still require new codes and scores. Per-vector scratch
and code allocations remain in clustering; the bulk serving path already reuses
scratch. Measure whole-phase costs before implementing this separate change.

`tq1-encoding-profile-01` completes that single phase probe. The private overlay
passes MC19, actual factory/library bindings and exact dad381 FiQA model bytes.
Root preparation totals1.008105 s: descriptor storage.296804 s, bundled query
encoding.692964 s, other work.01834 s. Root training assignments total6.927002 s
and final full-corpus assignment3.240993 s. Matching scoring totals10.075639 s
out of10.167994 s (99.09%); center encoding totals.012823 s and new root-query
encoding is zero because the cache is reused. Root scoring is the measured
bottleneck; this does not cover all serving encoders or establish a whole-build
encoding fraction.

There are14,020 inner calls/140,200 inner assignments, encoding21,984,251 query
vectors and18,245,280 changing-center vectors. Raw inner phase walls overlap;
query encode60.840 s, center encode97.169 s and scoring118.904 s cannot be added
to build wall or called CPU shares. Encoding includes FHT, allocation, norms
and packing. The instrumented30.052 s build is not a speed comparison. Reuse of
immutable input-query codes remains an exact candidate; changing-center codes
and distances still need recomputation. No encoding rewrite is applied.

- Existing assignment kernels are useful: TQ4 and TQ1 whole-build improvements
  were 2.146x and 4.241x within each mode, preserving saved trees and recall.
  These do not establish that quantized training matches float quality.
- The old ArguAna inner-only screen found no overall L2/PrefixDoubling win:
  float/IP reference 38.391 s, recall@10 .946875 at 32 probes; IP PrefixDoubling
  50.640 s/.946094; L2 uniform 36.763 s/.932813. Outer objective was always IP.
  A fresh `git ls-remote` on November 12 confirms upstream `main` is still
  `048e1ed6ff956febe80874c30f2178119ef104e4`, the revision already inspected.
- Current FiQA float and TQ8 baselines exceeded the bounded 240 s/180 s screens
  at 16 workers. Both are incomplete: no completed build time or quality may be
  inferred. FiQA has 57,638 documents/7,314,136 vectors/d128. The user's target is
  under 10 s at 192 threads; the linear 16-thread equivalent is 120 s, but must not be
  presented as a measured scaling result.
- Exact inner-Lloyd stopping passed bit-exact center and empty/zero-mass oracles
  in `early-stop-inner-screen-01`. Of 12 real weighted/unweighted IP/L2 cases,
  only weighted IP/k64 stopped at 9 rather than 10 updates. Tracking cost produced no
  consistent gain. Reject this candidate for production on current evidence.

## Outer and inner objective screen

`arguana-objective-screen-01` completes the previously missing outer-objective
comparison. Four independent builds use seed 0, float assignment, uniform inner
initialization, weighted means, leaf size 100, five outer updates, 16 Parlay
workers and one OpenMP/BLAS worker. Serving is always IP/MaxSim with TQ4 centers,
TQ1 leaves, TQ8 reranking of 50 candidates, and Wards compression at .5. Recall
uses the same 128 queries (sample seed 42) at k=10.

| Outer / inner | Build wall, s | Build user CPU, s | Recall, probes 16 / 32 |
|---|---:|---:|---:|
| IP / IP | 23.416 | 367.77 | .9015625 / .9468750 |
| L2 / IP | 74.742 | 406.27 | .9015625 / .9468750 |
| IP / L2 | 22.699 | 356.61 | .8867188 / .9328125 |
| L2 / L2 | 26.968 | 423.97 | .8453125 / .9046875 |

IP/IP and L2/IP saved indexes are byte-identical (SHA256
`c02fa1a957dad46bdadd10f89d2b551ccc0e54474997a39616dec6905d5b2d92`), as are
their predictions at both budgets. With unit spherical centers, squared L2 and
negative IP differ by an assignment-independent term and scale. This explains
the equivalence here; zero/nonunit centers and floating-point ties can break
that equivalence on other inputs. All 61,188 center vectors in this IP-inner
tree are unit length within 1.17e-7, with no zero centers. L2 inner means are not unit-normalized, so
center norms affect assignments and the trained tree. Both inner-L2 recipes
lose quality in this screen. Do not pursue those recipes on FiQA.

The outer-L2/IP wall outlier is not a demonstrated 3.2x algorithm penalty:
user CPU rises only about 10.5%, and host contention changes across the runs.
These are one-build quality screens, not controlled speed measurements.

## e-means

Bessa et al., [e-means: outperforming k-means (in one-tenth the epochs)](https://arxiv.org/abs/2611.09501), November 6, 2026. Official CPU implementation:
[Faiss pinned source](https://github.com/facebookresearch/faiss/tree/2b648f0605f2ef3d14c4705350f8a0b214463878/contrib/e_means).
The copied research reference retains its MIT license and exact commit. Its
spherical mode supports IP, but its CPU API has no per-point weights. Published
large-problem speedups cannot be transferred to our 400–2560-row inner calls.

`emeans-inner-screen-01` uses real leaf-vector proxies, same initial centers,
n=20k, d128, k20/64/128, unweighted IP/L2, three repeats. Timing includes setup,
training and nearest-final-center reassignment. Official labels from the moving
last pass are not silently treated as nearest-final-center labels.

| IP k | Local Lloyd10, ms | Official e2, ms | Official e3, ms |
|---:|---:|---:|---:|
|20|1.782|20.540|18.387|
|64|6.291|24.679|35.999|
|128|9.122|34.757|50.799|

Official e-means improves held-out negative-dot objective in all three cases
(for k128: -.732303 Lloyd versus -.735526/-.739048), but is slower in this CPU
reference. `emeans-inner-single-01` repeats only IP/k20/k128 with all worker
pools at 1: Lloyd 2.075/21.828 ms, e2 16.783/35.057 ms, e3 14.695/50.936 ms.
The quality improvement warrants one small weighted C++ EMA experiment with
reusable assignment matrices. It must be labeled an adaptation, not official
e-means, unless all reference components are implemented and verified.

The independent `weighted-ema-screen-01` C++ adaptation passes weighted
Lloyd-limit, weight-scaling, nonunit/zero input and final-assignment oracles.
It reuses at most 256 rows of Eigen score storage for every method, so the
statistical batch size does not give one method an artificial score-cache
advantage. Setup, training and exact nearest-final-center assignment are timed;
all methods use one worker and inverse-document-length weights. Three repeats
produce bit-identical centers. Held-out negative-dot objective uses the total
held-out weight as its denominator (lower is better).

| k | Flat Lloyd10 ms / held-out | EMA2 ms / held-out | EMA3 ms / held-out |
|---:|---:|---:|---:|
|20|1.625 / -.624386|.934 / -.630842|1.205 / -.631860|
|128|12.378 / -.728101|4.671 / -.734573|5.361 / -.736674|

EMA2 is 1.74x/2.65x faster than the same-storage control on these proxies and
improves their objective. Flat Lloyd10 agrees with saved legacy weighted
centers within 2.98e-8. The saved legacy timing uses a different worker budget
and excludes final assignment, so it is not used for a direct speed ratio.
This warrants a bounded whole-index experiment, not default adoption: a better
inner objective need not give better multivector retrieval recall.

`ema-integration-build-01` passes disabled-path coarse and float-training byte
fingerprints against the frozen runtime, plus weighted/norm, initialization,
zero/one-sample, reproducibility and ownership oracles. The full build retains
the existing sample, seed, float assignments and historical default initializer.
`arguana-ema-screen-01` and `arguana-ema-full-queries-01` give:

| Inner trainer | Build, s | Full-query recall, probes 16 / 32 |
|---|---:|---:|
| Legacy Lloyd10 | 20.636 | .909175 / .943954 |
| EMA2 | 18.474 | .904196 / .940896 |
| EMA3 | 18.684 | .907539 / .942034 |

The custom implementation's proxy gains translate to about 10% faster builds,
but not equal retrieval recall. EMA3 loses 23/27 of 14,060 correct neighbors at
probes 16/32. Pause further variants while center-size and quantized-refinement
experiments have higher priority. Do not change the production default or ship
`core/emeans` solely on the inner-objective result. All custom source, MIT
attribution, compiler inputs and runtime artifacts remain in the result archive.

The full-batch EMA-kernel control is not storage-equivalent to legacy Lloyd:
legacy reseeds truly empty groups and emits zero for nonempty zero-mass groups,
while the control retains previous zero-mass centers. Its spherical readout and
reduction arithmetic also differ. EMA additionally revives low-mass groups.
The stronger positive-weight/all-zero-corpus revival policy must be tested before
promotion; the current zero oracle covers zero rows with zero total mass.

## Quantized outer assignment refinement

`refinement-build-01` passes 1,364 oracle rows covering IP/L2, weighted means,
general norms, zeros, ragged clouds, ties and TQ1/TQ4. With refinement disabled,
coarse labels are byte-identical to the pinned baseline DSO. R=1 preserves
coarse labels; scoring every center recovers float argmin. A final float pass
preserves the learned centers exactly. The standalone oracle needed an explicit
core DSO link; the recorded retry reused the already compiled candidate.

`arguana-refinement-screen-01` uses the same serving/query/worker policy as the
objective screen, with TQ1 inner and outer assignment and historical default
inner initialization. Each variant retains all ten inner updates.

| Variant | Build, s | Recall, probes 16 / 32 |
|---|---:|---:|
| TQ1 baseline | 16.358 | .89140625 / .93203125 |
| TQ1 top-2 float refinement | 17.149 | .90468750 / .93593750 |
| TQ1 final float partition | 17.141 | .89218750 / .93125000 |

Top-2 refinement adds 167,040 float cloud scores across 234 assignment calls,
changes 4,323 coarse labels during training/partitioning, and improves sampled
recall for about 5% more elapsed build time in this screen. It remains below
float's 32-probe recall, so do not promote it as an equal-quality default.
One final float partition has no useful quality gain; stop that branch.

`quantized-refinement-build-01` adds TQ4/TQ8 shortlist scoring with cached root
queries, one center encoding per assignment, and codec dispatch outside row
loops. Independent scalar bit/panel decoding, R1/R2/all-center labels, reordered
root rows, weights, norms and float-anchor fingerprints pass. The full-build
float anchor matches the prior saved tree exactly. The screen controller's
search-path construction bug was repaired; its completed anchor was reused,
with the original failure and recovery controller retained.

| Top-2 refiner, inner3 | Build, s | Peak RSS MiB | Sample recall, p16 / p32 |
|---|---:|---:|---:|
| Float | 15.730 | 1635 | .903125 / .938281 |
| TQ4 | 16.333 | 2067 | .907813 / .938281 |
| TQ8 | 16.465 | 1822 | .899219 / .929688 |

TQ4/TQ8 refinement adds asymmetric 8-bit query caches (TQ4 does not imply
4-bit queries). Root extra-query preparation is .112/.123 s; cumulative
center encoding across calls is 1.872/.671 s, respectively. These per-call
totals are diagnostic and can overlap in parallel execution. TQ8 is slower
and lower quality here; stop it. TQ4's sampled low-probe gain does not persist:
`arguana-tq4-refinement-full-01` gives .905477/.936558 on all queries, below
float refinement's .907468/.941110 with greater build cost and memory. Stop
that branch too. Float remains the preferred tested shortlist refiner.

## Fewer inner updates

`arguana-work-reduction-01` changes only the fixed inner budget from ten to
three; weights, sampling, initializers, five outer rounds and serving remain
fixed within each pair. Float uses explicit uniform initialization; TQ1 uses
historical default initialization. The two references are identified separately
and are not presented as initialization-controlled float/quant comparisons.

| Candidate | Reference / candidate build, s | Candidate recall, probes 16 / 32 |
|---|---:|---:|
| Float, inner3 | 23.416 / 19.497 | .900000 / .937500 |
| TQ1 top-2, inner3 | 17.149 / 16.018 | .903125 / .938281 |

Float saves about 17% elapsed but loses .009375 recall at 32 probes. Quantized
top-2 saves about 7% elapsed, with a small sampled recall change of -.001563 /
+.002344. These separate shared-host observations do not establish paired
speed ratios. The modest reduction also shows that the seven removed inner
updates are not the only major full-build cost. Do not keep reducing iterations
blindly. Evaluate the promising quantized tree on all queries before FiQA.

`arguana-full-query-training-01` reuses the four existing trees, with no rebuild,
and scores all 1,406 queries (14,060 ground-truth neighbors):

| Training | Recall, probes 16 / 32 |
|---|---:|
| Float, inner10 | .908962 / .943954 |
| TQ1, inner10 | .899929 / .937909 |
| TQ1 top-2 float, inner10 | .908179 / .938407 |
| TQ1 top-2 float, inner3 | .907468 / .941110 |

Top-2 refinement's low-probe gain persists. The inner3 variant loses 21/40
correct neighbors versus float at probes 16/32 (.001494/.002845 recall), while
its earlier build screen took 16.018 versus 23.416 s. This is a candidate tradeoff,
not an equal-quality default or a serving-speed claim. Additional model seeds
and matched-recall serving comparison remain necessary before adoption.

## ANNS Lloyd assignment

`anns-inner-assignment-screen-01` tests the real k128/n2560/d128 weighted proxy
after one weighted center update. Dense Eigen and Vamana search receive the
same moved centers; previous assignments warm-start the searches. Every ANN
sample includes center storage, one graph-building pass and all assignments.
Input construction and weighted mean generation are outside both clocks.
All paths use one worker and three samples.

| Assignment | Median ms | Exact-label agreement |
|---|---:|---:|
| Dense | 3.226 | 2560 / 2560 |
| ANNS R8/L8 | 13.977 | 2312 / 2560 |
| ANNS R16/L16 | 27.713 | 2460 / 2560 |

Independent double scoring confirms positive objective excess for both ANN
variants. At these small inner center counts, graph/search overhead exceeds
the dense assignment it replaces, while assignments are less accurate. Stop
this branch before whole-index training. This does not reject ANNS for much
larger codebooks. The historical trainer's bool `two_pass` argument no longer
matches the overlay's integer `num_passes` constructor; the experiment uses an
explicit one-pass graph build rather than accidentally requesting zero passes.

## Remaining bounded gates

1. Document-size-dependent budgets are now screened on ArguAna and a bounded
   Quora whole-document subset with exact subset ground truth (results below).
   The global half-average/aligned policy fails Quora's serial gate. Keep the
   automatic default; do not assume64 or query length is universal.
2. Compress stored tree centers after training on an existing tree. This remains
   prepared-only while the simpler size change is being confirmed. Leaf document
   compression is a later gate: existing Wards emits vector sums, which cannot
   enter the unit-norm TQ1 leaf scorer without an explicit normalization policy.
3. A second bounded FiQA attempt (`fiqa-refine2-inner3-01`) also exceeded 120 s
   without completing; no recall or finished build time exists. Stop larger runs
   until a stronger small-data candidate appears. The target remains 192 threads,
   with no measured scaling claim from these 16-worker timeouts.

The current L2 recipes have no demonstrated MVIVF benefit. The public API
selection is retired and validated in `ip-api-validation-01`. Private Euclidean
k-means remains used by PQ/FastScan codebook training. Avoid new public experimental knobs until
measurements justify keeping an algorithm.

## Center-cloud sizes

`center-sizes-build-01` compiles only the frozen build facade. Invalid schedules
fail before parallel construction; unset schedule produces the exact saved
auto-average tree. Actual baseline sizes are root 94 clouds × 140 vectors and
339 deeper clouds of 121–150 vectors: 61,188 center vectors in 434 nodes.
Private per-level schedules are experimental sidecars, not a new public API.

`arguana-center-sizes-01` uses float/default-init/IP, weighted legacy inner10,
five outer iterations, seed 0, and the same 16-worker policy. Each mixed schedule
runs only after a useful fixed-size result.

| Size policy | Build s | Center vectors | Sample recall, p16 / p32 |
|---|---:|---:|---:|
| Average | 20.636 | 61,188 | .902344 / .946875 |
| Fixed64 | 11.240 | 28,416 | .903906 / .935156 |
| Fixed128 | 18.383 | 56,064 | .908594 / .939844 |
| Root192 / deeper64 | 23.860 | 40,896 | .903125 / .936719 |
| Root64 / deeper192 | 13.584 | 73,216 | .871094 / .921094 |

The root-heavy mixed schedule costs more without improving recall; the
deeper-heavy schedule loses substantial recall. Stop both. Fixed64 reduces
build user CPU from 322.82 to 173.51 s as well as wall time; the roughly 1.84x
build ratio is a same-host screen, not a clean compilation/scaling result.

`arguana-center-sizes-full-01` checks all 1,406 queries. Fixed128 reaches
.910597/.943528 at probes 16/32, close to average's .909175/.943954. Fixed64
reaches .903983/.938762 at 16/32, then .947866 at 40 probes. That higher-probe
point permits a useful matched-quality serving comparison.

`arguana-size64-matched-serving-01` runs one balanced ABBA block with two fresh
processes per setting/mode, all queries, identical frozen serving code, Wards
query .5 and TQ8 rerank50. Latency uses one worker and batch uses 16. Prediction
files are byte-identical across each setting's serial/batch repetitions.

| Metric | Average, p32 | Fixed64, p40 |
|---|---:|---:|
| Full recall@10 | .943954 | .947866 |
| Serial mean ms | 2.866 | 2.531 |
| Serial p95 ms | 3.608 | 3.115 |
| Serial p99 ms | 3.925 | 3.404 |
| Batch QPS | 4,971 | 5,500 |

Serial mean falls 11.7%, p95 13.7%, and p99 13.3%; batch improves 10.6% at
higher recall. This shared-host result is promising, not universal acceptance.
Single-query
latency is the deciding gate; a batch gain cannot excuse a latency regression.
Normalized center-only Wards compression is prepared but unexecuted. Keep it
behind confirmation of this simpler trained-size change; leaf compression is
still design-only, preserving original corpus ownership for final TQ8 reranking.

`arguana-size64-seed1-01` independently builds both settings with model seed 1
and confirms the result: build 21.253→11.485 s (CPU 335.40→180.08 s); recall
.942532 at auto/p32 versus .945164 at fixed64/p40. Serial mean falls
2.835→2.497 ms, p95 3.575→3.098 ms, p99 3.967→3.405 ms; batch QPS rises
4,689→5,453. Both build peaks stay about 1.36–1.37 GiB. The gain is primarily
time and center-vector storage, not a large peak-memory reduction.

`arguana-size32-01` tests one smaller setting. It builds in 7.466 s with 13,888
center vectors, but p40 recall .937980 misses the average/p32 target .943954.
P64 reaches .953770; balanced serial mean is 2.865→2.843 ms, p95
3.592→3.522 ms and p99 3.903→3.777 ms, with batch QPS +6.45%. The small mean
difference amid process variation is weak speed evidence compared with the
repeated fixed64 result. Stop the absolute-size grid.

Quora metadata supports the dataset-dependence question: `beir/quora_old` has
522,931 documents and 8,140,678 vectors, with mean 15.57 and median 14 vectors
per document (99.928% below 64). Its query file is padded to 32 vectors, which
does not determine the capacity needed by a center summarizing many documents.
The full corpus is still 3.89 GiB, so the next pilot uses whole documents from
a bounded subset and proves its top-10 ground-truth completeness explicitly.

## Document-scaled and aligned center budgets

`center-ratio-build-01` preserves the exact automatic tree with factor1 and
alignment1. Its experimental half-average/alignment16 rule uses the full local
document-cloud mean, rounds to the nearest positive multiple of16 (ties up),
and respects explicit positive sizes. On ArguAna this produces root64 and
deeper64/80; on short-cloud Quora it mostly produces16. These are frozen
experimental facade changes, not public options or production defaults.

An initial oracle incorrectly assumed the representative count could not
exceed the number of input vectors. Historical fixed-slot PointCloudSet
construction repeats real input directions when necessary. The corrected
oracle proves all16 representatives come from the two actual input directions
and preserve their deduplicated MaxSim. Failure and correction are archived;
the implementation did not change to satisfy the oracle.

`arguana-center-ratio-01` uses all 1,406 queries and balanced serving processes:

| Metric | Automatic, p32 | Half-average/aligned16, p40 |
|---|---:|---:|
| Build seconds | 20.704 | 11.272 |
| Center vectors | 61,188 | 30,528 |
| Recall@10 | .943954 | .946728 |
| Serial mean ms | 2.827 | 2.480 |
| Serial p95 ms | 3.710 | 3.061 |
| Serial p99 ms | 4.238 | 3.365 |

Batch elapsed ratio is .8684, about15.2% more QPS. This supports the same
ArguAna opportunity as fixed64; it does not establish a universal policy.

`quora-center-ratio-fixture-01` uniformly samples32,768 whole documents with
seed42 (509,623 vectors; mean15.552, median14) and512 queries. Original top2000
neighbors contain at least97 sampled documents for every selected query; each
sampled top10 is separated from the source cutoff (minimum margin .0433123).
Thus its exact subset top10 is complete, rather than treating truncated source
neighbors as arbitrary subset truth. Raw coordinates, sorted original-ID maps,
source hashes and independent double-MaxSim spot checks are retained. Results
are subset recall, not full522,931-document Quora recall.

`quora-center-ratio-01` rejects the global policy:

| Metric | Automatic, p32 | Half-average/aligned16, p40 |
|---|---:|---:|
| Build seconds | 4.748 | 4.563 |
| Center vectors | 30,477 | 30,432 |
| Recall@10 | .648047 | .668750 |
| Serial mean us | 437.834 | 465.104 |
| Serial p95 us | 572.488 | 613.646 |
| Serial p99 us | 710.265 | 774.340 |

Small clouds barely save center-panel work, and extra probes cost more than
they save. Batch was skipped after the serial failure. The bounded same-probe
check `quora-center-ratio-same-probe-01` gives candidate p32 recall .641406,
below baseline .648047; no extra timing grid followed. Keep the historical
automatic policy globally. The next section records the center-only compression
test; it also fails its quality gate.

## Post-training tree-center compression

`tree-compression-build-02` compiles one frozen facade, preserving the private
ISA implementations. Independent normalization/cancellation/owner oracles and
unchanged-default prediction fingerprints pass. `arguana-tree-compression-gated-01`
loads the existing float tree, compresses each center cloud once with Wards0.5,
normalizes merged directions, and saves an immutable candidate. Topology,
document membership, original leaf data and TQ8 reranking remain unchanged.
The no-merge and cancellation fallback preserve original directions.

Compression takes .0594 wall seconds/.866 CPU seconds with16 workers and cuts
61,188 center vectors to26,444 (root13,160→6,008; deeper48,028→20,436).
On the fixed128 queries, automatic p32 recall is .946875. The compressed tree
reaches .927344 at p32 and .936719 at the only recovery p40. It fails quality,
so serial/batch expansion and full-query confirmation are skipped. No .7 or
fresh-training grid followed. This policy is rejected; smaller trained
representatives remain the stronger measured ArguAna result. No production
default or source changed. Leaf compression remains design-only.

## Exact TQ1 leaf-code duplicates

`tq1-duplicates-counts-01` checks full serving-code bytes within each document,
using the same immutable TQ1 model on the original unit corpus. ArguAna has
415 duplicate codes among 1,208,012 vectors; FiQA has 3,334 among 7,314,136.
After actual 16-vector panel padding, deduplication would save only 7,168 bytes
(0.035%) and 55,552 bytes (0.045%), respectively. Direct-encoding and independent
integer-minimum oracles pass, but both miss the predeclared 3% payload-saving
gate. Stop before implementing compaction. No leaf kernel, raw corpus, TQ8
reranking, training behavior or production default changed. The counts, document
histograms and timing/provenance are retained in that result directory.

## FiQA fixed64 transfer screen

`fiqa-size64-01` completes one fresh float/weighted/IP legacy-inner10 build
with five outer iterations, leaf100, sample limits100/20, seed0 and16 unpinned
workers. Fixed64 builds in81.861 s (1,280.50 user CPU s,12.85 system s), with
7.051 GiB peak RSS. This is a finished16-worker measurement, with no192-thread
timing or extrapolated sub10-second claim.

Serving uses all648 queries, TQ4 centers/TQ1 leaves/TQ8 rerank50, Wards0.5.
The reference is the existing historical default-size FiQA tree; the current
frozen serving runtime is identical for both. Its old build time is excluded.

| Metric | Retained default, p32 | Fresh fixed64, p40 |
|---|---:|---:|
| Recall@10 | .574228 | .578086 |
| ABBA serial mean ms | 3.424 | 2.542 |
| Serial p95 ms | 4.604 | 3.532 |
| Serial p99 ms | 5.449 | 4.089 |
| Serial peak RSS GiB | 4.874 | 4.740 |
| Batch QPS, one process each | 5,705 | 7,021 |

Serial mean falls25.8% with higher recall; both tail percentiles improve.
The two baseline process means vary3.756/3.092 ms, so retain the raw shared-
host variation. Single/batch/repeated prediction bytes match within each tree.
Batch numbers are the quality-screen samples, not balanced throughput acceptance.

Historical training source/build realization and corpus content hash cannot
be recovered completely from that skeleton; exact row coverage/config/GT
compatibility does not prove its source/corpus provenance. This establishes a
useful practical recipe candidate, not a causal center-size comparison or a
paired build-speed ratio. No global default changed.

`fiqa-size64-same-source-01` makes exactly one fresh default-size attempt with
the same frozen runtime, source, inputs, recipe and requested16 workers,
retaining the existing S64 tree. S0 reaches its300-second cap without an index
or completed build JSON. It stops with zero retries and zero searches. CPU is
2,311.36 user plus9.79 system seconds, about7.72 active cores on average versus
15.67 during S64. Different effective concurrency prevents claiming a pure
3.66x size-policy wall-time gain; default quality/latency remain unavailable.
The attempted peak6.03 GiB is not completed-index peak memory. The practical
historical-tree comparison above remains the only completed FiQA quality screen.
