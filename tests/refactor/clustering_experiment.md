# MVIVF training-quality experiment

This experiment changes construction while serving remains IP with the same
center/leaf codecs, query compression, reranking and search grid. It is separate
from `parity.py` and `compare.py`, whose shared original skeleton isolates serving
implementation differences. A shared skeleton cannot measure a different
clustering algorithm's index quality.

The first phase fixes outer and inner assignment quantization to float. Keep the
corpus, outer seeding, weighting, sampling caps, centroid count, outer iteration
budget and serving representation fixed. Starting from `same/default/legacy`
inner training, change one field per parent-child comparison: either the inner
objective or initialization, or an explicitly selected implementation. A second
change needs its own comparison against that parent recipe; comparing only the
combined change to the original would confound the causes. Explicit `uniform`
initialization establishes the same IP sampling normalization used by the new
initializers. Compare IP/L2 objectives and `kmeans++`/`prefix_doubling` against
that explicit-uniform parent, and compare `adaptive` against its own otherwise
identical parent. The reference retains the original default behavior.

For each recipe, use the same set of outer model seeds (at least three for a
headline conclusion) and independent build repetitions. Build a new skeleton
for every recipe/seed/repetition. Search that recipe's own skeleton. Fix the
query-sampling seed independently from the model seed and retain the exact
selected query rows. Reverse recipe order on alternating blocks to reduce
ordering bias. Run each build and search schedule in a fresh process with
identical physical-CPU affinity, NUMA first-touch placement and thread settings.

Record the exact upstream source commit and local changes for any imported
UltraKMeans code. An upstream implementation name is not enough provenance.
Native module and every resolved shared library must remain identical throughout
one experiment; changing implementation code starts a new run. Changes made
through validated runtime construction options remain explicit recipe fields.

Construction records include index build wall time, process peak/resident bytes,
artifact size, tree height and leaf-size diagnostics. Also retain loading and
serving preparation times because quantized representations and TQ8 reranking
may account for work outside the centroid loop. Optional `diagnostics` invokes
standalone `cluster_point_clouds` on a fixed sampled corpus in another fresh
process. It records initialization, outer assignment and centroid-update stage
timers. The last stage includes grouping, sampling, inner Lloyd assignment and
float updates; the API does not expose those components separately. Verbose
float-cost reporting and final assignment add work outside the stage timers.
This standalone workload also lacks recursive MVIVF root encoding reuse, so its
total time must not replace production tree build time.

Search measurements use the complete predeclared probe/rerank grid for every
recipe and seed. Retain all repetitions, raw predictions, native and Python-call
times, per-query latency percentiles and optimized batch QPS. Report comparable
recall using only observed points: for each reference point and paired seed,
find the fastest candidate achieving at least the reference recall minus a
stated tolerance. Do not extrapolate missing high-recall points. Report both
identical-budget recall changes and the matched-recall speed ratios, plus their
variation across seeds/build repetitions. Lower build time alone does not
establish a better index.

After inspecting the float-assignment results, select a training recipe explicitly
for the second phase. Rebuild that recipe using float, TQ8, TQ4 and TQ1 assignment,
with all other fields fixed. The serving metric/codecs remain unchanged. Retain
a fresh float reference in the same run; earlier timings may come from a
different machine state. Quantized assignment always keeps float centroid
updates and the same weighting/norm semantics.

The output root is `/ssd2/kishen/mvsic_new`. Before any native work, honor
`/home/kishen/MVSIC/emdivf_experiments/artifacts/EMDIVF_BENCHMARK_ACTIVE.json`;
while it is active only prepare plans and inspect source. A run must stop if
that marker becomes active. No candidate is selected or accepted before these
measurements exist, and VNNI/AMX remain unvalidated on this host.

`configs/clustering_base.json` fixes the serving policy and query sample;
`configs/clustering_experiment.json` declares a three-seed float study, explicit
parent-child variants, and single-query/eight-worker batch schedules. The CPU IDs are examples
from one NUMA node on the development host. Check `lscpu -p=CPU,CORE,SOCKET,NODE`
and the available affinity, then edit both build and search CPU lists before
running. The controller validates every generated benchmark config before a
build and rejects an unavailable CPU set. Use a small pilot grid first, then
predeclare denser probe/rerank points where the recall envelope needs resolution.

An installed wheel is required for execution; every build/search subprocess runs
outside the checkout with `PYTHONPATH` cleared. Schema-only planning can use the
source package and does not load native code:

```sh
PYTHONPATH="$PWD" python tests/refactor/build_quality.py \
  --config tests/refactor/configs/clustering_experiment.json --plan-only \
  --output /ssd2/kishen/mvsic_new/build-quality/arguana-plan-001

python tests/refactor/build_quality.py \
  --config tests/refactor/configs/clustering_experiment.json --phase training \
  --output /ssd2/kishen/mvsic_new/build-quality/arguana-training-001
```

Use a fresh output directory for each invocation. `builds.json`, `searches.json`
and optional `profiles.json` retain raw records. `summary.json` contains paired
construction speed/memory ratios, identical-budget recall changes and observed
matched-recall QPS ratios. `plan.json` retains exact recipes, controller/config
hashes and ordering. Native extension/shared-library content, corpus hashes and
query sampling must remain stable throughout one run.

After reviewing those results, copy the experiment config and add
`"selected_training": "<chosen recipe name>"`. The controller deliberately has
no implicit winner. Run that new config with `--phase quantization` and another
fresh output directory. Its float/TQ8/TQ4/TQ1 recipes differ only in assignment
quantization; each receives new paired builds and serves its own tree.

## Limited ArguAna pilot

`configs/clustering_arguana_pilot.json` uses the complete ArguAna corpus (8,674
documents, 1,208,012 d128 vectors) and 128 fixed queries selected with seed 42.
It matches the native serving pilot: leaf size 100, five outer iterations,
weighted inner updates, TQ4 centers/TQ1 leaves, Wards 0.5, root many-to-many,
TQ8 reranking of 50 candidates, and probes 4/8/16/32. Build and batch search use
eight physical cores, CPUs 0/4/8/12/16/20/24/28, on NUMA node 0; single-query
latency uses CPU 0. The CPU list was checked against this host's topology and
available affinity. Recheck it before running elsewhere.

The primary objective remains **IP retrieval**: outer assignment and serving use
negative directed mean MaxSim, and recall is against the same IP ground truth.
`inner_metric="same"` therefore means spherical IP inner training. The alternative
`inner_metric="l2"` changes only the inner objective to squared L2 with weighted
arithmetic means. L2 centroids retain their lengths; the experiment must not
normalize those means or substitute L2 ground truth. The source corpus and
original query vectors retain the norms required by TQ1 serving. Assignment
quantization in the second phase changes both outer and inner scoring and keeps
float centroid updates; it never changes the serving representation.

| Recipe | Parent | Single changed field |
| --- | --- | --- |
| reference | — | `same/default/legacy` |
| ip_uniform | reference | `inner_init=uniform` |
| l2_uniform | ip_uniform | `inner_metric=l2` |
| ip_kmeanspp | ip_uniform | `inner_init=kmeans++` |
| ip_prefix | ip_uniform | `inner_init=prefix_doubling` |
| ip_adaptive | ip_uniform | `inner_backend=adaptive` |
| l2_adaptive | l2_uniform | `inner_backend=adaptive` |
| ip_kmeanspp_adaptive | ip_kmeanspp | `inner_backend=adaptive` |
| l2_prefix | l2_uniform | `inner_init=prefix_doubling` |
| l2_prefix_adaptive | l2_prefix | `inner_backend=adaptive` |

The pilot has one model seed (0), one independent build per recipe, two search
schedules, four operating points per schedule and three timing repetitions:
10 builds and 80 measured operating points. Python-call timing and standalone
stage diagnostics are disabled to keep this first screen small. It does not
establish seed robustness or justify a default change. Explicit initializers use
the model seed; the historical reference retains its original seeding behavior.

Run only after the original/refactor native pilot and any compilation finish.
The released EMD marker does not reserve the machine. Keep the installed wheel
and native libraries unchanged throughout both phases:

```sh
MVSIC_CPU_ISA=avx512 /tmp/mvsic-refactor-env/bin/python tests/refactor/build_quality.py \
  --config tests/refactor/configs/clustering_arguana_pilot.json --phase training \
  --output /ssd2/kishen/mvsic_new/refactor/arguana-training-pilot-001
```

Inspect paired build time, peak RSS, identical-budget recall and observed
matched-recall latency/QPS. Tolerance 0.001 on only 128 queries allows roughly
one recovered neighbor at k=10; it is an explicit matching tolerance, not a
statistical confidence bound. If the probe envelope cannot reach a parent's
recall, report that missing match and extend the same grid for every affected
recipe before choosing a winner.

Then copy this pilot config, set `selected_training` explicitly to the reviewed
recipe, and run `--phase quantization` in a fresh output directory. This creates
four new paired builds (float/TQ8/TQ4/TQ1) and 32 operating points. Do not select
a recipe merely to make that phase runnable. The full
`clustering_experiment.json` retains three seeds, 1,000 queries, a wider
probe/rerank grid and separate diagnostics for confirmation of any promising
change. Report selected configuration and scope alongside every result.
