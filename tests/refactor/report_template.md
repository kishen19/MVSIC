# MVIVF refactor validation record

Status: pending measurements. This template contains no performance claims.

- Original source revision and local diff hash:
- Refactor source revision and local diff hash:
- Original/refactor executable and extension SHA256:
- Compiler/version, flags, dependency pins, ISA selection:
- Compile scope: original IP TQ4-center/TQ1-leaf specialization versus new supported codecs/ISAs:
- CPU model, affinity, NUMA policy, worker count, coordination idle window:
- Dataset/query/ground-truth SHA256, selected query rows, dimensions, normalization:
- Shared original skeleton SHA256 and build/assignment settings:
- Independently built original/refactor skeleton SHA256 for every ABBA trial:
- Warmup, repeat count, ABBA blocks, timing scope and summary statistic:

| Measurement | Original | Refactor | Scope/interpretation |
|---|---:|---:|---|
| Clean compile wall seconds / peak process RSS | pending | pending | Same flags/jobs; state subset vs full extension |
| Public-header incremental wall / peak RSS | pending | pending | Original uses isolated source snapshot |
| Implementation incremental wall / peak RSS | pending | pending | List affected translation units |
| Binary/extension size, import time | pending | pending | Record dependencies and subset scope |
| Index build seconds / peak RSS | pending | pending | Same build settings; retain both skeletons |
| Load/prepare + first query seconds | pending | pending | Include lazy TQ8 preparation for both |
| Native p50/p95/p99 latency | pending | pending | Report each tree scope separately, matched recall |
| Python p50/p95/p99 API latency | pending | pending | Includes native and output conversion |
| Native/Python batch QPS | pending | pending | Optimized batch, same threads/affinity |
| Prepared/final/peak resident bytes | pending | pending | Includes data/runtime/allocator/scratch |
| Recall@k and scoring work | pending | pending | State tie policy and score agreement |

Report both `shared_original` and `independent_build` scopes. The first isolates
serving changes; the second must establish recall and throughput for each
implementation's own fresh trees. Do not pool their measurements or match recall
across scopes. Per-query latency arrays are retained in raw results; calculate
percentiles explicitly because the comparison aggregate summarizes QPS and RSS.

Attach full per-operating-point results, observed matched-recall curves, original
and refactor predictions, and all raw repeats. State any slower or lower-recall
points and the evidence from investigating them. Do not use a faster point at
lower recall as acceptance evidence.

Correctness coverage: raw IP oracle, rejection of retired L2 retrieval, all serving
codecs, independent IP/L2 clustering objectives and assignment codecs, single/batch
and root-M2M on/off, exact/TQ8 reranking, query compression,
noncontiguous IDs, owner lifetime, PCS mmap/copy, malformed save/load, rebuild,
root-leaf/collapsed topology, and installed package/CLI outside the repository.

Hardware coverage: AVX2 and AVX-512 actual-host paths; VNNI and AMX are not hardware
validated on this host. Record unsupported paths explicitly.
`MVSIC_CPU_ISA` controls serving/clustering; RaBitQ rotation dispatch remains
hardware selected. Native comparison excludes its separately reported first
query even with zero warmup; installed zero-warmup measurements include it.
