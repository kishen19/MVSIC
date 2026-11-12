# MVIVF serving discrepancy audit — November 11, 2026

This read-only source/result audit launches no compilation or experiment. The
original checkout remains untouched. Runtime changes are not yet accepted as
resolving the wide-thread regression.

## Observed discrepancy

`queued-serving-192-01/attempt-001` completed six balanced original / before /
candidate processes, 25 batches each, on the same saved original tree and 1,024
queries. All-25 process median throughput is 18,897 original, 17,590 before and
17,951 candidate queries/s. The last-ten summaries are 19,011 / 17,659 / 17,630.
The guard completed without interference. The gap persists after warmup; it
cannot be dismissed as a single noisy timing. Two processes per implementation
still limit statistical confidence.

## Concrete source differences

| Difference | Current source | Evidence and implications |
|---|---|---|
| Serial validation of every query | `mvivf_search.cc:197` | Original batch entry has no corresponding norm scan. Existing 96-core profile measures 2.68–3.01 ms in validation. Keep the checks, parallelize independent query validation, and preserve caller-side exceptions and lowest-row error precedence. |
| Serial destruction of prepared query ownership | `mvivf_search.cc:221` and end of batch scope | The vector destroys every owning query on the caller. Existing profile measures 1.96–1.99 ms. Original codec and compression temporaries use Parlay sequences, whose nontrivial element destruction runs in parallel. Original center queries also move into worker-local storage during probing. |
| Serial probe-vector cleanup | `mvivf_search.cc:230` | About 0.35 ms in the same profile. |
| Serial grouping and candidate offsets | `mvivf_search.cc:235–245` | About 0.62–0.66 ms; original Parlay grouping logs about 0.36 ms in the corresponding 96-core run. Preserve disjoint deterministic candidate ranges when optimizing. |
| Padding validity check on every TQ4 panel | `turboquant_mv.h:229–256` | Corrected IP epilogue compares scales and masks minima even for full real panels. Only a trailing partial panel needs exclusion. A future full-panel/tail split must retain real zero vectors and the current arithmetic order. Cost is not isolated by the saved profile. |
| Extra per-leaf pointer conversion | `mvivf_search.cc:252–255`, `mvivf_kernels.cc:217–220` | The facade builds generic query pointers, then the backend builds typed pointers. Original builds typed pointers directly. Small allocation/copy overhead; no measured attribution. |

Stage medians describe diagnostic 96-core runs, not a prediction of a 192-thread
speedup. Summing independent medians is not a causal performance measurement.
The serial validation and owning-object cleanup are the strongest measured
targets; the remaining items have lower priority.

Root M2M retains the same fuse-and-score algorithm. TQ8 reranking retains the
same numerical kernel and nested candidate parallelism; the refactor removes
several old allocations. Fixed candidate offsets remove atomic scatter and
sentinel compaction, and TQ1 omits redundant per-leaf sorting. There is no source
evidence that the new facade simply performs more TQ8 scoring per query.

## Output and work equivalence

Performance-only changes to validation scheduling or independent object
destruction should preserve current IDs and float bits exactly. Query values,
compression, traversal, leaf scoring, selection and reranking remain unchanged.
Use correctly constructed owning containers; Parlay uninitialized storage is
appropriate only for fully written trivial buffers here. Exceptions must not
escape the homegrown scheduler's worker callbacks.

Previous and current refactor prediction files are byte-identical throughout
the retained 192-thread study. Historical `coarse-cutoff-01` pairs have identical
visited document ID-to-score-bit maps for every one of their 128 sampled queries
on each of two fixed trees: 62,184 and 66,103 visited document scores respectively.
These are leaf candidate comparisons, not total center-distance counts.

The original has different tie rules: pointer-based tree heap ordering,
score-only replacements, negative packed-score ID ordering in single search,
and arrival-order-dependent `nth_element` cutoff membership in batch search.
The refactor supplies stable score/row-ID ordering. A coarse tie at rerank50 can
therefore change final neighbors even when all shared scores agree exactly.
The original also lacked the AVX-512 IP padding exclusion. These correctness
changes prevent blanket output identity with the original on all possible inputs.

Original 192-thread stdout reports 626,224 distance comparisons in all 26 batch
observations of both original processes. Current batch output has no equivalent
center/leaf counter. The historical cross-tree Python work diagnostics used
only the refactor engine. Consequently, total work equivalence on the full
1,024-query workload has not been demonstrated. A temporary diagnostic should
record per-query center counts, visited leaf IDs/sizes, leaf comparison totals,
and rerank candidate IDs on both engines before interpreting this gap further.

## Next controlled changes

1. Close the full-batch work-count and remaining cutoff-tie classification gaps.
2. Change prepared-query ownership cleanup independently; require byte-identical
   predictions and counts against the current validated snapshot.
3. Parallelize boundary validation separately, preserving first-error semantics.
4. Measure each change with the same fixed workload at the requested 192 threads.
5. Consider grouping/pointer and TQ4 tail-only-mask changes only if a residual gap
   warrants them. Keep training/assignment experiments separate from serving.

Artifacts live under `/ssd2/kishen/mvsic_new/refactor/`. Relevant records include
`batch-stage-profile-001`, `candidate-query-final-01`, `coarse-cutoff-01`,
`original-batch-cutoff-final-01` and `queued-serving-192-01/attempt-001`.
