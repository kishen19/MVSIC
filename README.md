# MVSIC
MVSIC (pronounced "music") stands for Multi-Vector Search, Indexing and Clustering, a library and benchmarking tool for multi-vector clustering and retrieval.


<hr>

### Build
`bazel build ...`

### Build MVIVF Indices
- Update the data path and results path in index.sh
- Update params like `maxsize`, `iters`, etc.
- Run `bash experiments/index.sh <index> <dataset>`

### Computing Stats
- Update the data path and results path in stats.sh (and stats_gold.sh)
- Update params like `maxsize`, `iters`, `k`, etc.
- Run `bash experiments/stats.sh <index> <dataset>`
