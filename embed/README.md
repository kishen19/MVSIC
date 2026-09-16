# Embedding datasets for MVSIC

Embed BEIR, LoTTE and ViDoRE retrieval datasets into the complete set of files used by MVSIC: multi-vector embeddings in csr, stable ID mappings, gold GT, and exact Chamfer neighbors GT.

## Commands

Run commands from the repository root:

```bash
# BEIR: downloads the source data into data/beir/scifact when needed.
python -m embed beir scifact --compute-gt

# LoTTE: dev is the default split; search is unsuffixed and forum uses -forum.
python -m embed lotte pooled --compute-gt
python -m embed lotte pooled --split test --compute-gt

# ViDoRe with ColPali.
python -m embed vidore arxivqa \
  --source vidore/arxivqa_test_subsampled_beir --compute-gt
```

BEIR and LoTTE use `colbert-ir/colbertv2.0`. ViDoRe uses `vidore/colpali-v1.3-hf`. Use `--checkpoint` to override either default.

LoTTE is loaded through `ir_datasets`, which exposes the official domain, pooled, dev/test, and search/forum hierarchy. A LoTTE invocation generates the corpus once and generates both search and forum queries.

## Output convention

Each dataset lives at `data/{beir,lotte,vidore}/<dataset>/`. For the default split and query kind:

```text
<dataset>_points.pcs
<dataset>_point_ids.txt
<dataset>_queries.pcs
<dataset>_query_ids.txt
<dataset>_gold_neighbors.csr       # when labels exist
<dataset>_chamfer_neighbors.gt     # with --compute-gt
<dataset>_manifest.json
```

LoTTE defaults to `dev/search`. Forum query artifacts use `-forum`; test artifacts use `-test`. Thus the four query stems are `pooled`, `pooled-forum`, `pooled-test`, and `pooled-test-forum`. The dev and test corpora are different; forum and search within the same split share their corpus.

Queries with available gold GT are retained only when at least one judged document exists in the generated corpus. Filtering happens before query encoding, so query embeddings, query IDs, gold GT, and exact GT always use the same row order.

## Dependencies

The lightweight format and exact-GT modules need NumPy. Dataset/model-specific dependencies are imported lazily:

- BEIR: `beir`, `colbert-ai`
- LoTTE: `ir-datasets`, `colbert-ai`
- ViDoRe: `datasets`, `transformers`, `accelerate`, and PyTorch
