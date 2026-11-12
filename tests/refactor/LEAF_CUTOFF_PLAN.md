# Paired leaf-95/100 construction and retrieval

This uses existing `compare.py`; no additional framework or default change is
needed. Do not execute until root validates the final combined native build and
reserves the measurement window. Use those final binaries, not earlier binaries
paired with a newer source identity. Keep their archived compiled-source manifest.

```sh
/tmp/mvsic-final-env/bin/python tests/refactor/compare.py \
  --config tests/refactor/configs/arguana_leaf100_final.json --quiet-host \
  --output /ssd2/kishen/mvsic_new/refactor/arguana-leaf100-final-01
/tmp/mvsic-final-env/bin/python tests/refactor/compare.py \
  --config tests/refactor/configs/arguana_leaf95_final.json --quiet-host \
  --output /ssd2/kishen/mvsic_new/refactor/arguana-leaf95-final-01
```

The JSON files differ only in `leaf_size`. Both implementations get both
settings. Each run builds original/refactor/refactor/original with eight pinned
node-0 workers (0,4,8,12,16,20,24,28), float assignment, IP serving/default inner
objective, unchanged weighting, seed 0, five outer iterations and unchanged
centroid parameters. Each independent artifact is then searched on CPU 80 with
one worker at probes 14/16/18/24/32, rerank 50, TQ8, Wards 0.5, k=10 and root-M2M
enabled. Every one of the 1,406 queries is used, identically across builds.

Together this is **eight fresh builds and 40 native search processes**, two
independent builds per implementation/leaf setting and three warm timing
repetitions per search process. Original nondeterministic construction remains
visible as a distribution. Each search loads its matching built artifact; no
shared-tree search grid or separate full native pilot is duplicated. Expect
roughly **12–15 minutes**, using observed ~39-second builds and single-worker
search/preparation durations; the larger probe points can extend it. The raw
build peak RSS, search peak RSS, preparation/first-use costs, all repetitions,
predictions, exact-recall curve and input/runtime identities are retained.

The existing comparison reports original-versus-refactor matched recall with
zero tolerance within each leaf setting. Compare the two runs jointly to assess
leaf 95 versus 100 for each implementation, preserving the two build outcomes
instead of selecting a fortunate seed/tree. Report both fixed-budget recall
deltas and faster points at equal-or-higher recall. The two leaf settings run
sequentially; guard and process variability remain relevant, and a reverse
leaf-order repeat is warranted only if the conclusions are close or unstable.

The reason for 95 is structural: two new root groups of sizes 96 and 98 remain
leaves at 100 whereas overlapping original groups of 103 and 111 split into
eight leaves each. A 95 cutoff targets these transitions using an existing
explicit parameter. It also changes other nodes and adds center work; it may
lose recall or speed and is **not** a dataset-independent default recommendation.
The 89-document crossing is intentionally untouched by this bounded test.

After this study, use fixed-tree batch scaling rather than rebuilding again.
The temporary `/tmp/mvsic_python_endpoints.py` driver covers the installed API
without new builds: probes 8/32, one-worker latency/eight-worker batch, original
03 shared tree, all 1,406 queries, original/refactor/refactor/original order,
16 Python worker processes and three repetitions. It pins the subset original
module, installed refactor module, all MVSIC DSOs and the existing Python timing
worker. Default invocation prints the plan; add `--execute` only after approval
of the next timing window. Example after final module validation:

```sh
/tmp/mvsic-final-env/bin/python /tmp/mvsic_python_endpoints.py \
  --original build/refactor-clang/_mvsic_baseline.cpython-312-x86_64-linux-gnu.so \
  --refactor /tmp/mvsic-final-env/lib/python3.12/site-packages/mvsic/_mvsic.cpython-312-x86_64-linux-gnu.so \
  --output /ssd2/kishen/mvsic_new/refactor/python-endpoints-01
```

Add `--build-provenance /path/to/final/native-source-manifest.json` when available.
This measures Python API calls including native work and result conversion; it
does not relabel them native-only latency. Expected additional time is roughly
2–4 minutes. Do not use the old broad Python-followup config, which also rebuilds
indices and duplicates a large native grid.
