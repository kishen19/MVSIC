# MVIVF refactor checkpoint evidence

Status: correctness verified; performance acceptance remains incomplete.
All results below are under `/ssd2/kishen/mvsic_new/refactor`.
No commit or push is authorized or has been made.

## Measurements

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

The latest source is archived in
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
- Default build time/memory above precede the latest Parlay changes; a final
  construction confirmation remains outstanding.
- Latest source has not yet received a refreshed installed-wheel/C++-consumer
  check and GCC validation. Those passed for the preceding final-build-04
  snapshot (`final-wheel-03`, `cpp-install-05`, `gcc-final-validation-02`).
- Clean original/refactor compilation-cost comparison is not complete. The
  latest incremental native build took162.70 seconds with four compiler jobs;
  it is not a clean-build acceptance measurement.
- VNNI remains compile-only; AMX is not implemented or hardware-validated.

The staged tree is a reviewable work-in-progress checkpoint. It must not be
described as having satisfied the user's unconditional "at least as good as
before" criterion until the remaining evidence is collected.
