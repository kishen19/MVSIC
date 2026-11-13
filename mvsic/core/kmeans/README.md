# K-means implementation

`lloyds_kmeans.h` contains MVSIC's float-centroid Lloyd's driver and its float
and TQ8 backends. The compiled TQ4/TQ1 adapters and outer cloud assignment live
in `algorithms/mvivf/mvclustering/mvclustering_kernels.cc` (relative to `mvsic/`).
`kmeans_util.h` retains the wrappers used by existing indexing and quantization
methods.

The headers under `lloyds/`, `lloyds_weighted/`, and `seeding/` are the minimal
transitive source subset needed from
[ParAlg/Parallel-KMeans](https://github.com/ParAlg/Parallel-KMeans), revision
`6cdb5b8649ea254c31a329fcdbbf6537e01452b2` (the `mvsic` branch, February 25, 2026).
They were copied from that Git revision, not from a moving branch. The source
repository is not required at configure, build, or install time. The original
relative header paths are retained so the untouched original MVIVF baseline
can use the same pinned sources for comparisons.

The copied subset consists of:

- `lloyds/{anns,kmeans,multi-swap,pairwise}.h`
- `lloyds_weighted/kmeans_weighted.h`
- `seeding/{kmeans_utils,kmeansparallel,kmeansplusplus,ksetcover,prefixdoubling,uniformlyrandom,utils}.h`

The copied headers retain their original contents except for provenance
comments and `inline` on the header-defined `symmetrize()` in `ksetcover.h`.
That existing MVSIC dependency patch prevents multiple-definition errors when
several translation units include the trainer; it does not change its
algorithm. No upstream drivers, benchmarks, or Ward clustering sources are
included.

## Attribution and licensing

The pinned source revision has no root README, LICENSE, or CITATION file, and
none of these twelve headers contains a copyright or license notice. No
license has been inferred or assigned to them. Any future upstream licensing
information should be recorded here before redistributing these sources under
a stated license. The source repository and exact revision above identify the code provenance.

## Paper

Xiaojun Dong, Kishen N Gowda, Jessica Shi, MohammadHossein Bateni, Laxman
Dhulipala, Xiaoen Ju, and Jakub Łącki. **UltraKMeans: Fully Scalable and High
Quality k-means.** PPoPP 2027, accepted, to appear.
[Author publication page](https://kishen19.github.io/). No DOI is available yet.

## Experimental inner trainer

The local trainer also adapts selected ideas from upstream `main` revision
`048e1ed6ff956febe80874c30f2178119ef104e4` (June 10, 2026), fetched into an
isolated checkout. The original vendored headers above remain at their pinned
revision for baseline comparisons. `lloyds_kmeans.h` identifies the adapted
sections and retains float centroid updates.

`MVClusteringConfig` provides two independent controls:

| Field | Default | Experimental choices |
| --- | --- | --- |
| `inner_init` | `default` (historical seeding) | `uniform`, `kmeans++`, `prefix_doubling` |
| `inner_backend` | `legacy` | `adaptive` |

MVClustering uses IP for inner training, outer assignment and serving.
Private L2 k-means remains available for Euclidean PQ/FastScan codebooks;
it is not a public MVClustering option. Explicit IP initialization samples directions
using nonnegative squared chord distances and produces unit initial centers
(zero vectors remain zero). It retains raw input lengths for assignments and
weighted float updates. K-means++ and PrefixDoubling sampling include document
weights when weighted inner training is enabled. Explicit initializers use
`config.seed`; the historical default retains its original seed sequence.

The adaptive float backend uses flat input storage, avoiding a huge-page
allocation for each small inner problem, streaming float centroid reductions,
and Eigen distance tiles sized for a roughly 4 MiB working set. Quantized
assignment uses its codec kernels independently of this option; the TQ8 backend
already has flat input and streaming means. TQ4 and TQ1 assignment use packed
centers and AVX2/AVX-512 kernels, with TQ1 norms applied before minima so general
norms remain valid. The historical float layout and 256-row tiles remain the
default.

Distance-based seeding keeps its incremental nearest-distance arrays. For
float assignment only, it also records assignments to the exact initial float
centers and hands them to the first Lloyd update. IP assignment scores are
computed from raw vectors separately from chord sampling distances, so zero
vectors and nonunit inputs keep their metric semantics. Scalar seeding and
Eigen assignment can round near ties differently; tests compare objective
values with a tolerance rather than requiring identical labels near ties.
Quantized assignment always scores its own encoded centers. Every centroid
update invalidates old distances; no triangle bound or stale assignment cache
is reused. Immutable TQ8 input bytes, norms and rotations are shared between
outer cloud assignment and inner vector assignment.

## Measured training choices

The controlled ArguAna pilot compared ten independent float-assignment builds
with fixed IP serving. L2 inner training, alternative IP initializers and the
adaptive backend did not establish a better build/recall tradeoff across the
observed grid. A subsequent four-cell outer/inner IP/L2 screen found identical
trees when the inner centers were spherical, and lower retrieval recall for L2
inner means. MVClustering therefore exposes IP only. The default remains float
assignment with `inner_init="default"` and `inner_backend="legacy"`.

The subsequent assignment optimizations preserved exact serialized trees and
recall: TQ4 whole-build time fell from 82.989 to 38.664 seconds, and TQ1 from
131.505 to 31.005 seconds on the controlled AVX-512 run. These are improvements
within each assignment mode, not evidence that quantized training matches float
quality. At 32 probes, float recall@10 was .946875, TQ4 .93828125, and TQ1
.93203125. TQ8's lowest-center-ID correction changed its trained tree, so its
before/after comparison is not a packing-only speed measurement.

These measurements use one model seed, one build per snapshot and 128 fixed
queries. They support the measured optimizations but do not justify changing
training defaults. See `REFORMAT_CLUSTERING_HANDOFF.md` at the repository root
for configurations, full curves, artifact hashes, hardware scope and result
locations. No VNNI or AMX path was hardware-validated.
