# EMDIVF

EMDIVF selects training-query point clouds by farthest-first traversal under
exact EMD, stores a top-B Chamfer document-ID list for each prototype, and
searches the union of a few routed lists. This implementation supports the
inner-product Chamfer setting, with the same normalized ColBERT point clouds
used by the existing experiments. It is a C++ library and command-line runner;
Python bindings and the paper benchmark registry are not added here.

The standard search path is TQ4 Chamfer Vamana routing over prototypes, TQ8
Chamfer prototype reranking, one-copy TQ1 document scoring, and the shared TQ8
Chamfer final reranker. Wards query compression defaults to threshold 0.5 for
routing and candidate scoring; both TQ8 reranks use the original query. Exact
EMD is used during prototype selection. It uses Euclidean vector ground cost
and uniform mass over actual vectors: assignment for equal cloud sizes and
balanced transport for unequal sizes.

Every bucket stores document row IDs, and buckets can overlap. The corpus-wide
TQ1 panels and TQ8 reranking representation each live once. Candidate IDs are
deduplicated with a bitset, enumerated in physical order, and scored in roughly
128 KiB chunks with an 8 KiB prefetch lookahead. The first 512 bytes of a future
document are prefetched. No replicated bucket panels or experimental scan
policies are required.

## Build and run

Run from the repository root. The artifact directory in these examples is the
gitignored local SSD symlink described in
[emdivf_experiments](../../emdivf_experiments/README.md); another writable output
directory works equally well. The code and Bazel targets do not depend on that
symlink or on external experiment sources.

```bash
bazel --output_user_root=/ssd1/kishen/cache/.cache/bazel build \
  //mvsic/emdivf:bench_build //mvsic/emdivf:bench_search

out=emdivf_experiments/artifacts/indexes/main/fiqa

PARLAY_NUM_THREADS=192 bazel-bin/mvsic/emdivf/bench_build select \
  --train data/beir/fiqa/fiqa_queries_train.pcs \
  --state "$out/centers.state" --m 2048 --seed 42 --bounded 1

PARLAY_NUM_THREADS=192 bazel-bin/mvsic/emdivf/bench_build buckets \
  --train data/beir/fiqa/fiqa_queries_train.pcs \
  --docs data/beir/fiqa/fiqa_points.pcs \
  --state "$out/centers.state" --m 2048 --b 512 \
  --mode tq8-refine --shortlist 1024 --batch 16 \
  --cache "$out/corpus_tq8.bin" --journal "$out/buckets.journal" \
  --output "$out/buckets.idx"

PARLAY_NUM_THREADS=192 bazel-bin/mvsic/emdivf/bench_search graph \
  --train data/beir/fiqa/fiqa_queries_train.pcs \
  --index "$out/buckets.idx" --graph "$out/prototypes.graph" \
  --R 32 --L-build 200 --alpha 1.2

PARLAY_NUM_THREADS=24 bazel-bin/mvsic/emdivf/bench_search search \
  --train data/beir/fiqa/fiqa_queries_train.pcs \
  --docs data/beir/fiqa/fiqa_points.pcs \
  --queries data/beir/fiqa/fiqa_queries.pcs \
  --gt data/beir/fiqa/fiqa_chamfer_neighbors.gt \
  --index "$out/buckets.idx" --graph "$out/prototypes.graph" \
  --b 512 --k 10 --shortlist 128 --nprobes 8,16,32,64,128 \
  --num-rerank 100,200,500 --query-compress wards \
  --compression-threshold 0.5 --timing-queries 128 \
  --output emdivf_experiments/artifacts/results/main/fiqa.csv
```

Selection checkpoints and bucket journals resume interrupted construction.
Increasing M reuses the same selection prefix. The TQ8 corpus cache is shared
between bucket builds. `tq8-refine` scans the corpus with TQ8, then float-scores
the shortlisted documents; it is approximate globally. `--mode exact` uses a
full float Chamfer scan instead. Changing B, input files, seed, or shortlist
requires a separate journal/cache as directed by the provenance checks.

Search initializes its encodings once per process. Quality is evaluated over
every query; latency is measured sequentially with one search worker over the
first `--timing-queries` rows. Zero, the default, times all rows. CSV output
includes Recall@k, unique candidates, prototype distance comparisons, stage
times, latency percentiles, and QPS. The CSV records `k` and `recall_k`; the
`recall_10` compatibility column is populated only when `--k 10`.
Ground truth is the exact Chamfer neighbor format written by `embed/`, rather
than relevance qrels. Dev queries and their corresponding exact GT should be
used for tuning.

## C++ API and formats

Link `//mvsic/emdivf:emdivf` and include `mvsic/emdivf/emdivf.h`:

```cpp
emdivf::CloudSet train("train.pcs", true), docs("docs.pcs", true);
auto buckets = emdivf::Load("buckets.idx");
emdivf::SearchIndex index(train, docs, std::move(buckets), "prototypes.graph");
emdivf::SearchParams params;
params.nprobes = 32;
params.num_rerank = 500;
auto result = index.Search(query, params);
```

`emd.{h,cc}` contains the transport kernel, `index.{h,cc}` contains the prototype
and bucket index plus its original exact builder, and `emdivf.{h,cc}` contains
graph construction and the standard quantized search path. `bench_build.cc`
provides resumable selection and batched bucket building; `bench_search.cc`
provides graph building and evaluation.

Existing `EMDIVF1` bucket indexes remain readable. Newly written routing graphs
use version 2 and record the exact prototype order, which is checked against
the bucket index. Historical version-1 Chamfer graphs remain readable, but they
contain no prototype IDs: callers must ensure their prototype order matches.
Graph node IDs are prototype positions; bucket IDs and returned IDs are corpus
row positions. Training query rows are stored separately in `Index::centers`.
When loading an old index that stores 2,048 documents per bucket, pass
`--b 512` explicitly to reproduce the standard setting. The default zero uses
the full stored bucket size.

The existing EMD regression tests are available as
`//mvsic/emdivf:emd_test`. This migration was reviewed statically while the
machine was busy; neither compilation nor tests nor benchmarks were run for
the migration. Historical experiment numbers are summarized in
[EXPLORATION.md](../../emdivf_experiments/EXPLORATION.md).
