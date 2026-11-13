# Development-host comparison plans

`arguana_native_pilot.json` uses the configured `build/refactor-clang` drivers, the
ArguAna corpus and exact ground truth, and 128 sorted query rows selected with
seed 42. Construction uses float assignment, model seed 0 and eight workers.
Serving uses IP, TQ4 centers/TQ1 leaves, Wards 0.5, root many-to-many, TQ8
reranking of 50 candidates, and probes 4/8/16/32. Native latency and batch use
one and eight workers on physical CPUs 0/4/8/12/16/20/24/28, all on NUMA node 0
on this host. No experimental inner-training options are enabled.

Retrieval and MVClustering are IP-only in both comparison drivers. Rebuild the
drivers after public API changes: the
original comparison now instantiates one IP TQ4-center/TQ1-leaf variant, so earlier
two-metric compilation measurements are a different scope. The original source
repository is unchanged.

Both original and refactor executables/modules must be built with Clang 15.
The installed refactor wheel is built from that same tree. The initial GCC
diagnostic tree is excluded: original in-class
specializations require the compiler used by the original Bazel build. Its GCC
compile failure is not a performance result or a refactor-specific regression.
Do not mix a GCC-built refactor with the Clang-built original in these plans.

The pilot contains 16 operating points, each measured in both shared-original
and independent-build scopes with ABBA ordering: four build processes and 128
native search processes, with three timing repetitions per process. Start only
after the drivers pass correctness checks and the root build/debugging activity
has finished; a released EMD marker alone does not make simultaneous compilation
and benchmarking controlled.

```sh
MVSIC_CPU_ISA=avx512 /tmp/mvsic-refactor-env/bin/python tests/refactor/compare.py \
  --config tests/refactor/configs/arguana_native_pilot.json \
  --output /ssd2/kishen/mvsic_new/refactor/arguana-native-001
```

`arguana_python_followup.json` uses the same policies with probes 8/32 and adds
the original CPython 3.12 comparison module and the installed refactor module
from `/tmp/mvsic-final-env`. The older `/tmp/mvsic-refactor-env` snapshot remains
available for pre-optimization training comparisons. It runs a fresh native
comparison followed by matching Python API trials: four build, 64 native search
and 64 Python search processes. It is a later, separate run after both modules
import successfully. It measures Python API performance separately from the
installed-wheel correctness tests; it is not a Python-only replay.

```sh
MVSIC_CPU_ISA=avx512 /tmp/mvsic-refactor-env/bin/python tests/refactor/compare.py \
  --config tests/refactor/configs/arguana_python_followup.json \
  --output /ssd2/kishen/mvsic_new/refactor/arguana-python-001
```

Use new result directories for subsequent runs. An AVX2 comparison uses the
same config with `MVSIC_CPU_ISA=avx2` and a separately named result directory;
the original driver retains its host-native build in either run. The override
controls the new serving/clustering backend, while RaBitQ rotation selects its
own hardware ISA. These are pilots, not final high-recall acceptance studies:
broaden the query sample and probe/rerank envelope only after investigating the
initial curves. Compilation cost requires separate clean/incremental plans.

`clustering_base.json` and `clustering_experiment.json` belong to the independent
inner-training study documented in [clustering_experiment.md](../clustering_experiment.md).
They use eight build/batch workers and three model seeds. The smaller
`clustering_arguana_pilot_base.json` and `clustering_arguana_pilot.json` retain the
same IP parent-child recipes with one seed, 128 queries, probes 4/8/16/32 and
rerank 50. They measure float assignment first; after explicit recipe selection,
the quantization phase rebuilds float/TQ8/TQ4/TQ1 with fixed IP serving. Do not
run this construction study concurrently with the native serving pilot.

The completed historical ten-recipe pilot also included L2 inner objectives.
Those recipes and the metric API were retired after the L2 screens found no
benefit for MVIVF. The archived result directories retain their original plans.

Current work retains automatic representative sizing (`s: 0`). Fixed-size,
fractional-size and per-level sizing experiments are closed at the user's
request. Their measurements remain archived in the training study; no tuned
dataset-specific size recipes are shipped with this checkpoint.
