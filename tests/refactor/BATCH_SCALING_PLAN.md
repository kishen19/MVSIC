# Fixed-tree batch scaling protocol

This is a protocol only; no timing run is authorized by creating this document.
Root coordinates the measurement window after native correctness and installation
checks. Generated artifacts belong under `/ssd2/kishen/mvsic_new/refactor`.

The existing runner implements this protocol without creating new indices:

```sh
/tmp/mvsic-final-env/bin/python tests/refactor/cross_tree.py \
  --original build/refactor-clang/mvsic_baseline_bench \
  --refactor build/refactor-clang/mvsic_native_bench \
  --module /tmp/mvsic-final-env/lib/python3.12/site-packages/mvsic/_mvsic.cpython-312-x86_64-linux-gnu.so \
  --batch-scaling --blocks 1 --quiet-host \
  --quiet-window-note 'final validated binaries; reserved full-machine window' \
  --output /ssd2/kishen/mvsic_new/refactor/batch-scaling-01
```

Add `--plan-only` to inspect the 48-process first block without native work.
Default `--blocks 2` adds the reversed second block (96 processes); choose that
only if the reserved window permits it. `--scaling-workers 1,4,8` is a bounded
subset. Native comparisons must be rebuilt after the prepared-memory sidecar
change: scaling rejects binaries that omit this evidence. Estimated wall time is
4–6 minutes for the first block or 7–12 for both, subject to actual multi-core
preparation/scaling behavior. No construction cost is included.

Use the exact saved original-03 tree from `arguana-native-01` for both executables.
That tree was built with eight workers; keep that construction provenance fixed
and do not rebuild at each search thread count. Pin the original host-native
baseline, validated refactor executable and every dependent MVSIC DSO. Record
their build/source archive and compiler identities; do not hash a newer mutable
checkout and claim it is the compiled source. Compare original native AVX-512
with refactor forced AVX-512 first. AVX2 is a separate follow-up matrix after its
candidate passes correctness, not mixed into the primary scaling curve.

Hold serving IP, TQ4 centers, TQ1 leaves, TQ8 rerank 50, probes 16, k=10, Wards
0.5 and root-M2M enabled. Use two **fixed** batch sizes: the saved pilot 128-query
sample, then a separately saved sorted seed-42 sample of 1,024 of the 1,406 real
queries. Record each sample and ground-truth subset. The 1,024-query batch gives
96 workers enough work; repeating query rows to manufacture that batch is not
equivalent. Record predictions and recall for every trial, not only QPS.

Use fresh processes and existing native timing loops. Record lazy first-query
preparation separately, then one whole-batch warmup (the current native driver's
positive `warmup` means one batch), then five measured full-batch repetitions.
Retain every time, the selected repetition, process RSS and native QPS. Batch
elapsed time is not per-query latency; do not synthesize p99 latency from QPS.

## CPU and NUMA schedule

Use one logical CPU from each physical core at 1, 4, 8, 24, 48 and 96 workers.
The verified host has four packages/NUMA nodes with 24 physical cores each and
two SMT threads per core. The following nested schedule preserves CPU 80 from
the cross-tree diagnostic and keeps the first eight cores on node 0:

| Workers | CPU set |
|---:|---|
| 1 | 80 |
| 4 | 80,84,88,92 |
| 8 | 80,84,88,92,64,68,72,76 |
| 24 | Every CPU 0,4,8,…,92 (node 0) |
| 48 | Previous set plus 1,5,9,…,93 (nodes 0–1) |
| 96 | CPUs 0–95, one thread per physical core (nodes 0–3) |

Verify package/core/sibling/NUMA maps from sysfs and `lscpu` before launch rather
than assuming IDs on a different host. CPUs 80/84/88/92 have siblings
176/180/184/188; the guard reserves them too. Keep all thread-library environment
variables explicit and affinity identical for both implementations at each point.

The full-machine matrix requires the full-host quiet guard: `--quiet-host`, not
permission to overlap an external CPU 0–63 workload. A limited 1/4/8-worker
shared-host subset may be run separately with explicit authorization, but keep
its scope and results separate from the full-host scaling curve.

Do not silently change memory placement between paired implementations. Start
with the inherited Linux memory policy used by the existing driver. Each native
worker saves raw `.numa_maps` and `.proc_status` sidecars after its first query
prepares lazy caches and before warmup/timing. The latter includes
`Mems_allowed_list`; the controller hashes both. No memory placement is enforced.
At 48/96 workers, NUMA placement is part of this real serving
measurement, not pure arithmetic thread scaling. If placement differs enough to
explain a discrepancy, run a separately labeled paired `numactl --interleave`
control over the active nodes; retain both results rather than replacing the
unfavorable result. Corpus mmap page placement can persist across processes,
so include corpus/anonymous page placement evidence and do not drop host caches.

## Order and report

For each batch size and thread count, run original/refactor/refactor/original.
Then repeat the matrix in reverse thread-count and batch-size order. This gives
96 native processes (2 batch sizes × 6 counts × 4 ABBA entries × 2 blocks), four
process trials per implementation/cell. All paths use the same fixed tree; no
construction measurement belongs in this run. Allow a bounded first block if
the reserved window is shorter, explicitly reporting two trials per cell.

Report absolute QPS, median process-level QPS, min/max or dispersion, speedup
relative to each implementation's own one-worker point, efficiency QPS/(T×QPS1),
refactor/original ratio at each count, RSS and recall. Preserve per-cell background
guard samples and failures. Do not merge shared-host and full-host samples.

If 128 and 1,024 batches disagree, interpret root/grouped-leaf amortization and
available parallelism before changing dispatch. Test root-M2M off only at 1, 24
and 96 workers as a bounded separate diagnostic, not another full grid. A 192-SMT
point is worthwhile only if the 96-physical-core point is stable and has not
already saturated memory bandwidth; compare it to 96 on the same 1,024-query
batch and label SMT explicitly. It is optional, not a required acceptance point.
