# Controlled refactor comparisons

These are validation tools, separate from the installed benchmark interface. No
numbers in this directory are measured performance claims. Every heavy subprocess
checks `/home/kishen/MVSIC/emdivf_experiments/artifacts/EMDIVF_BENCHMARK_ACTIVE.json`
first. The controller also watches the marker while a command runs and terminates
its process group if the original experiment becomes active. Run only in an
agreed idle window; the marker check does not reserve the machine.

MVIVF retrieval supports IP only. L2 remains available for clustering; native
and installed API tests must reject an L2 retrieval request. Old IP/L2 parity
artifacts document earlier coverage and do not describe the current build scope.

For construction changes, including inner IP/L2 objectives, initialization,
adaptive kernels and assignment quantization, use
[`build_quality.py`](build_quality.py) and the
[training-quality experiment](clustering_experiment.md). That study independently
builds and searches each recipe's own tree. The original/refactor comparisons
below report both shared-tree serving and independent-build retrieval quality.

Compile `native_bench.cc` twice with the same compiler, optimization and dependency
versions:

* The refactor executable includes this checkout and links the compiled MVSIC
  library.
* The original executable defines `MVSIC_BASELINE=1`, includes
  `/home/kishen/MVSIC` and its dependencies, and uses the original headers directly.
  It must not link the new MVSIC library or include its core headers. It instantiates
  only the supported IP production TQ4-center/TQ1-leaf variant. No changes to the
  original checkout are needed.

This avoids hundreds of unrelated benchmark instantiations. Its compilation cost
is a **one-specialization IP baseline**, not the cost of the full original Python
extension or its full benchmark wrapper matrix. Record that scope in reports.
The original benchmark binaries found in the EMD experiment artifact folders
implement EMDIVF and are not substitutes for an MVIVF baseline.

For Python API comparisons, build `_mvsic_baseline` from `baseline_bindings.cpp`
and the original `mvsic/python/register_common.cc`, with original include paths.
It uses the original `BIND_INDEX` macros, Parlay list caster, GIL behavior, and
search signature for the same IP production variant. Compare it against the
new installed `_mvsic` extension using `python_bench.py`. Their return formats
differ by design: original Python lists versus new NumPy arrays. Record baseline
subset scope and both module hashes. Do not interpret a subset module's build
cost as the full legacy wheel's cost.

A comparison config looks like this (replace executable/module paths):

```json
{
  "original": "/ssd2/kishen/mvsic_new/build-original/native_bench",
  "refactor": "/ssd2/kishen/mvsic_new/build-new/native_bench",
  "points": "/ssd2/kishen/multivector/beir/arguana/arguana_points.pcs",
  "queries": "/ssd2/kishen/multivector/beir/arguana/arguana_queries.pcs",
  "ground_truth": "/ssd2/kishen/multivector/beir/arguana/arguana_chamfer_neighbors.gt",
  "metric": "ip",
  "leaf_size": 100,
  "niters": 5,
  "assignment": "float",
  "build_threads": 1,
  "build_cpus": [0],
  "cpus": [0, 1, 2, 3],
  "threads": [1, 4],
  "modes": ["latency", "batch"],
  "k": 10,
  "nprobes": [1, 2, 4, 8, 16, 32, 64],
  "rerank": [20, 50, 100],
  "root_m2m": [true, false],
  "compression": ["None", "Wards"],
  "threshold": 0.5,
  "tq8_rerank": true,
  "query_limit": 1000,
  "query_seed": 42,
  "warmup": 10,
  "repeats": 5,
  "build_blocks": 1,
  "search_blocks": 1,
  "search_scopes": ["shared_original", "independent_build"],
  "recall_tolerance": 0.001,
  "python": {
    "original": {"executable": "/tmp/mvsic-refactor-env/bin/python", "module": "/path/to/_mvsic_baseline.cpython-312-x86_64-linux-gnu.so"},
    "refactor": {"executable": "/tmp/mvsic-refactor-env/bin/python", "module": "/path/to/installed/_mvsic.cpython-312-x86_64-linux-gnu.so"}
  }
}
```

Choose available physical CPUs and a consistent NUMA node; CPU numbers above are
examples, not assumed topology. The controller applies the first `threads` CPUs
from `cpus` before allocation, so each job's first-touch placement follows the
same affinity. `build_cpus` controls construction. To compare cross-NUMA behavior,
make a separate explicit plan. Thread environment variables are set before native
or NumPy imports. Latency with more than one thread measures within-query
parallelism; batch measures the optimized batch traversal.

```sh
python tests/refactor/compare.py --config comparison.json \
  --output /ssd2/kishen/mvsic_new/comparisons/arguana-native-001
```

The output directory must be new and under `/ssd2/kishen/mvsic_new`. Binaries are
copied to hash-addressed immutable paths, so measurements do not follow mutable
`bazel-bin` symlinks. Resolved `libmvsic*.so` implementations are also copied and
hashed; an explicit `LD_LIBRARY_PATH` points to those copies, and loader resolution
is checked before measurements. Source executable path and hash, input hashes, query sample,
commands, CPU affinity, thread counts, stdout, resource usage, predictions, and
all native repetitions are retained. The controller uses sorted
`default_rng(seed).choice` query sampling, matching the legacy runner.

Each block runs original/refactor/refactor/original (ABBA), each in a new process.
Independent builds use identical parameters; all subsequent native and Python
searches are labeled with one of two scopes (both run by default):

* `shared_original`: both implementations load **the first original v4 skeleton**
  to isolate serving implementation differences. Each search block uses ABBA.
* `independent_build`: each implementation loads **each of its own newly built
  skeletons**, preserving the construction ABBA order, repeated for each search
  block. This measures default construction quality as well as serving speed.

The query sample, search parameters, affinity and worker count match across both
scopes. Python comparisons use the exact artifact from the corresponding native
trial. Every result retains the artifact hash and independent build number;
aggregation and matched-recall selection never combine the two scopes. A
shared-tree speedup alone does not establish quality of a freshly built index.
Search repeats report the minimum native time within each
process; aggregate comparisons use the median across the ABBA processes for
that implementation, scope and operating point. Set more blocks for noisy measurements. Native predictions from
the selected repetition and query latencies are saved, not only rounded QPS.

Both implementations initialize the TQ8 reranker lazily. The harness records load/prepare,
first-query, and subsequent warmup time separately. Compare `prepare_seconds +
first_query_seconds` when evaluating preparation plus the first request. Warm
latency excludes this first request for both implementations. Build time can
therefore include different serving preparation work; report the separate search
preparation and first-query measurements alongside it.

The native/Python comparison drivers always run that separately timed first
query, even when `warmup` is zero. The installed config runner with
`warmup_queries: 0` includes its first invocation in the first measured repetition.
Comparing numbers across those harnesses therefore requires matching and stating
the warmup policy; within each original/refactor pair the policy is identical.

`comparison.json` includes the measured operating points and observed
comparable-recall matches. For each original point, matching selects the fastest
measured refactor point attaining at least the original recall minus the explicit
tolerance, on the same mode/threads/compression/root/rerank-codec policy. Probe and
rerank budgets may differ. No interpolation or extrapolation fills missing recall.
Review both the matched-recall envelope and identical-parameter predictions;
results do not automatically certify the redesign. Investigate slower points,
recall loss, varying predictions across repetitions/processes, and memory growth.

Compiler measurements are optional ordered `compile_plan` entries in the same
JSON. They run before runtime comparisons; `--compile-only` omits runtime work.
For example:

```json
{
  "compile_plan": [
    {
      "name": "new-library",
      "phase": "clean",
      "command": ["cmake", "--build", "/ssd2/kishen/mvsic_new/build-new-clean", "--target", "mvsic", "--parallel", "1"],
      "threads": 1,
      "outputs": [
        "/ssd2/kishen/mvsic_new/build-new-clean/libmvsic.so",
        "/ssd2/kishen/mvsic_new/build-new-clean/libmvsic_core.so",
        "/ssd2/kishen/mvsic_new/build-new-clean/libmvsic_avx2.so",
        "/ssd2/kishen/mvsic_new/build-new-clean/libmvsic_avx512.so",
        "/ssd2/kishen/mvsic_new/build-new-clean/libmvsic_vnni.so"
      ]
    },
    {
      "name": "new-public-header",
      "phase": "incremental_public",
      "touch": "/home/kishen/MVSIC-new/MVSIC/mvsic/core/index.h",
      "command": ["cmake", "--build", "/ssd2/kishen/mvsic_new/build-new-clean", "--target", "mvsic", "--parallel", "1"],
      "threads": 1
    }
  ]
}
```

Use fresh build directories for clean phases, the same compiler/ISA/optimization
flags for both implementations, then separate public-header and implementation
changes. Baseline incremental header tests must use a `/tmp` source snapshot;
automatic touching of original checkout headers is rejected. Capture configure,
compile, link, and wheel commands separately when useful, and save their exact
flags/dependency pins. Disable compiler caches or state their warm/cold policy.
GNU `time` records wall/user/system time and maximum child-process RSS; with
parallel compiler jobs this is not aggregate simultaneous compiler memory.
Use one compile job for comparable peak-process RSS, and report job count.
Optional `wheel` phases can measure full new packaging, but the old subset
extension is not an equivalent full wheel workload.

The primary comparison tree is `build/refactor-clang`, using Clang 15 for both
implementations; its comparison targets are
`mvsic_baseline_bench`, `mvsic_native_bench`, `_mvsic_baseline` and `_mvsic`.
Use the resulting absolute executable/module paths in a comparison config after
the build succeeds. This diagnostic tree is not a clean-compilation measurement.
The initial `build/refactor` GCC tree is excluded from paired measurements:
original in-class specializations fail with GCC 12, whereas the original Bazel
configuration uses Clang 15. Keep the original source unchanged and compare
both implementations with the same compiler.
The new `mvsic` target builds all serving codecs across AVX2, AVX-512 and VNNI,
whereas the original driver instantiates one IP variant of TQ4 centers/TQ1
leaves for the native host. Report these different compile scopes explicitly.
The CMake comparison uses common dependency sources and optimization settings;
it is not a replay of every flag in the original Bazel build. Save
`compile_commands.json` and dependency provenance alongside compilation results.
Hash all five implementation DSOs as shown above: `libmvsic.so` alone does not
identify the ISA implementations. Runtime comparison also pins and hashes every
resolved `libmvsic*.so` dependency automatically.

`MVSIC_CPU_ISA` selects the serving/clustering specialization. RaBitQ's rotation
dispatcher independently selects an ISA from hardware capabilities, so an
`avx2` override is not a claim that every dependency executes only AVX2 code.

`report_template.md` lists the evidence needed for acceptance. VNNI and AMX cannot
be hardware validated on this machine; record those paths as compile/design-only.

For same-skeleton numerical parity before timing larger corpora, run the focused
helper against the same two native executables:

```sh
python tests/refactor/parity.py \
  --original /path/to/mvsic_baseline_bench --refactor /path/to/mvsic_native_bench \
  --cpus 0,1,2 --isa avx2 --isa avx512 \
  --output /ssd2/kishen/mvsic_new/parity-001
```

It generates a small normalized ragged d128 corpus, loads an original skeleton in
both implementations, and compares full score maps, partially visited sets and an
independent float64 negative-mean-MaxSim IP oracle. It covers exact/TQ8 reranking, single/batch
traversal and root-M2M settings; unsupported requested ISAs must fail explicitly.
The helper observes the same coordination marker and pinned-library rules.

The original AVX-512 TQ4 center scorer has a padding error for positive IP
distances, so partial-probe correctness comparisons use the untouched original
compiled for AVX2 as their reference. Its TQ1 batch fallback also explicitly
returns `FLT_MAX`; the failed original self-check is retained in `parity-02`.
For this reference, use the explicit single-query-only flag:

```sh
/tmp/mvsic-refactor-env/bin/python tests/refactor/parity.py \
  --original build/refactor-clang/mvsic_baseline_avx2 \
  --refactor build/refactor-clang/mvsic_native_bench \
  --reference-single-only --cpus 0,4,8 --isa avx2 --isa avx512 \
  --output /ssd2/kishen/mvsic_new/refactor/parity-ip-001
```

This records the excluded original batch path in provenance and still strictly
compares every new ISA/thread/single/batch/root path, including partial visited
sets, to original AVX2 single-query results. It does not allow refactor
mismatches. Without the flag, original batch self-checks remain enabled. The
original host-native AVX-512 executable remains the performance baseline;
these AVX2 reference runs make correctness claims only.

The bounded `cross_tree.py` diagnostic isolates the equal-recall slowdown seen
in the interrupted ArguAna pilot. It reads that pilot's original-00, original-03
and refactor-01 saved trees, verifies and pins them plus the exact 128-query
seed-42 sample, and runs both implementations on every tree. Two ABBA blocks
(the second reverses tree order) produce 24 one-worker processes, each with three
native repetitions at probes 16, rerank 50, TQ8 reranking and Wards 0.5. No index
is rebuilt. A separate untimed phase then records the public refactor
`search_with_stats` counters/stages per query; those timings are diagnostic only.

```sh
/tmp/mvsic-refactor-env/bin/python tests/refactor/cross_tree.py \
  --original build/refactor-clang/mvsic_baseline_bench \
  --refactor build/refactor-clang/mvsic_native_bench \
  --module /tmp/mvsic-refactor-env/lib/python3.12/site-packages/mvsic/_mvsic.cpython-312-x86_64-linux-gnu.so \
  --cpu 0 --quiet-host --quiet-window-note 'reserved window; root and training jobs stopped' \
  --output /ssd2/kishen/mvsic_new/refactor/arguana-cross-tree-01
```

Run only after rebuilding and validating both binaries and the installed module,
and after the training window ends. `--plan-only` performs no native work;
`--blocks 1` bounds the timing phase to 12 processes. The full schedule, compiler
commands and identities, source/input hashes, pinned libraries, every repetition,
predictions, recall and process memory are retained. Prediction differences are
reported per query and never automatically attributed to the known original
padding bug. Original build03 and refactor build01 receive their own exact
observed-equal-recall comparison; the lower-recall original build00 is not pooled
into it. The diagnostic alone does not establish independent-build acceptance.

`compare.py`, `build_quality.py` and `cross_tree.py` accept `--quiet-host` for
coordinated timing windows. It is off by default; the inner-assignment experiment
always uses the same guard. Every two seconds the guard records process CPU
deltas, excludes this controller and its descendants, and tolerates ordinary
background services. Two consecutive samples with any unrelated process using
at least 0.75 CPU cores, or unrelated processes totaling at least 1.5 cores, stop
our controller; its existing cleanup terminates only our child process group.
PID reuse cannot inherit an exclusion, and monitoring failures stop the run too.
`host_samples.jsonl`, `host_guard.json` and interruption records preserve the
evidence. The original coordination marker remains active independently. This
guard helps detect interference; it does not reserve the host or prove that a
short burst between samples had no effect.

When a limited-thread comparison must proceed alongside another job, pass
`--shared-host` instead of `--quiet-host`. The comparison and build-quality
controllers reserve the union of the CPUs in their actual build/search configs;
cross-tree and inner-assignment reserve `--cpu`. All SMT siblings are included.
An unrelated process is excluded from the contention pool only when the union
of **every thread's allowed CPU masks** is disjoint from those physical cores.
The same 0.75/1.5-core thresholds and two-sample rule apply to the overlapping
pool; unbounded affinity overlaps, and unreadable or unstable thread masks stop
the run. Full background CPU activity and thread masks remain in the sample log.

For example, use `cross_tree.py --cpu 80 --shared-host ...` on this machine:
CPU 80's sibling is 176, both outside the external job's CPU 0–63 mask. CPUs
84, 88 and 92 have siblings 180, 184 and 188 if a four-worker config is needed.
The guard verifies topology at runtime. Results explicitly carry scope
`shared_host_disjoint_cpu_affinity`; shared cache, memory bandwidth, power and
clock effects remain possible, so these results support the requested limited
comparison and must not be reported as isolated-host timing.

The same `cross_tree.py` runner supports a bounded probe-count comparison with
`--fine-probes`: original engine/original-03 tree at probes 16 versus refactor
engine/refactor-01 tree at probes 14, 15, 16 and 17. Each candidate has adjacent
original/refactor/refactor/original processes; the second block reverses candidate
order. The pilot uses the exact saved 128-query sample (32 timed processes).
Rerank, compression, quantizers, tree bytes and native timing policy stay fixed.
Fine-probe mode retains every repetition, raw prediction, exact recall numerator
and denominator, and matches only equal-or-higher baseline recall with **zero**
tolerance. It does not run the cross-tree stage-diagnostic phase.

Append `--fine-probes` to the cross-tree command above for that pilot. After
inspecting it, explicitly select promising candidates in a separate fresh output:

```sh
# Same --original/--refactor/--module and coordination arguments as the pilot:
python tests/refactor/cross_tree.py ... --fine-probes \
  --candidate-probes 15 --all-queries \
  --output /ssd2/kishen/mvsic_new/refactor/arguana-fine-probes-confirm-01
```

The confirmation uses all 1,406 original queries in file order and their full
ground truth. One candidate takes eight timed processes with default two blocks.
`--all-queries` requires an explicit candidate list; it never automatically selects
a setting from the pilot or changes an index default. `--plan-only` shows either
schedule without native imports or execution. Tree/probe combinations may return
different IDs intentionally; raw differences remain available for recall analysis,
and repeated predictions within each operating point are checked separately.
