# MVC
A library for multi-vector clustering algorithms

### Build
`bazel build ...`

### Build MVIVF Indices
- Update the data path and results path in index.sh
- Update params like `maxsize`, `iters`, etc.
- Run `bash experiments/index.sh <dataset>`

### Computing Stats
- Update the data path and results path in stats.sh (and stats_gold.sh)
- Update params like `maxsize`, `iters`, `k`, etc.
- Run `bash experiments/stats.sh <dataset>`
