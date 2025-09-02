index=$1
dataset=$2
reduced=""
if [ "$dataset" = "quora" ] || [ "$dataset" = "nq" ] || [ "$dataset" = "hotpotqa" ] || [ "$dataset" = "msmarco" ]; then
  reduced="_reduced"
fi

DATAPATH=/ssd2/laxman/multivector
RESULTSPATH=/ssd2/kishen/MVC

mkdir -p ${RESULTSPATH}/${dataset}/stats


maxsizes=(500)
iters=(5)
ks=(10 100)
cands=(1 2 4 8 16 32)
Rs=(128 256 512)
dfdes=(2560 5120 10240 20480)

if [ "$index" == "mvivf" ]; then
  for maxsize in ${maxsizes[@]}; do
    for iter in ${iters[@]}; do
      for k in ${ks[@]}; do
        bazel-bin/experiments/${index}_benchmark \
          -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
          -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
          -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
          -index ${RESULTSPATH}/${dataset}/indices/${index}_maxsize${maxsize}_iters${iter}.bin \
          -r ${RESULTSPATH}/${dataset}/stats/${index}_maxsize${maxsize}_iters${iter}_k=${k}.csv \
          -k ${k} \
          -npl 1 -npr 4096 -npmp 2 -npad 0
        echo ${index} ${dataset} ${maxsize} ${iter} ${k} done
      done
    done
  done
elif [ "$index" == "svh" ]; then
  for maxsize in ${maxsizes[@]}; do
    for cand in ${cands[@]}; do
      for k in ${ks[@]}; do
        prod=$((cand*k))
        bazel-bin/experiments/${index}_benchmark \
          -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
          -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
          -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
          -index ${RESULTSPATH}/${dataset}/indices/${index}_maxsize${maxsize}.bin \
          -r ${RESULTSPATH}/${dataset}/stats/${index}_maxsize${maxsize}_k=${k}_cand=${cand}.csv \
          -k ${k} -cands ${prod} \
          -npl 512 -npr 2048 -npmp 2 -npad 0 
        echo ${index} ${dataset} ${maxsize} ${cand} ${k} done
      done
    done
  done
elif [ "$index" == "vamana" ]; then
  for R in ${Rs[@]}; do
    for k in ${ks[@]}; do
      bazel-bin/experiments/${index}_benchmark \
        -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
        -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
        -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
        -index ${RESULTSPATH}/${dataset}/indices/${index}_R${R}.bin \
        -r ${RESULTSPATH}/${dataset}/stats/${index}_R${R}_k=${k}.csv \
        -k ${k} \
        -Ll 64 -Lr 2048 -Lmp 2
      echo ${index} ${dataset} ${R} ${k} done
    done
  done
elif [ "$index" == "muvera" ]; then
  for dfde in ${dfdes[@]}; do
    for k in ${ks[@]}; do
      bazel-bin/experiments/muvera_benchmark \
        -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
        -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
        -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
        -index ${RESULTSPATH}/${dataset}/indices/muvera_fde${dfde}.bin \
        -r ${RESULTSPATH}/${dataset}/stats/${index}_fde${dfde}_k=${k}.csv \
        -k ${k} -d_fde ${dfde} \
        -Ll 64 -Lr 2048 -Lmp 2
      echo ${index} ${dataset} ${dfde} ${k} done
    done
  done
elif [ "$index" == "muvera_norm" ]; then
  for dfde in ${dfdes[@]}; do
    for k in ${ks[@]}; do
      bazel-bin/experiments/muvera_benchmark \
        -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
        -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
        -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
        -index ${RESULTSPATH}/${dataset}/indices/muvera_norm_fde${dfde}.bin \
        -r ${RESULTSPATH}/${dataset}/stats/${index}_fde${dfde}_k=${k}.csv \
        -k ${k} -d_fde ${dfde} -norm \
        -Ll 64 -Lr 2048 -Lmp 2
      echo ${index} ${dataset} ${dfde} ${k} done
    done
  done
elif [ "$index" == "muvera_norm_norerank" ]; then
  for dfde in ${dfdes[@]}; do
    for k in ${ks[@]}; do
      bazel-bin/experiments/muvera_benchmark \
        -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
        -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
        -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
        -index ${RESULTSPATH}/${dataset}/indices/muvera_norm_fde${dfde}.bin \
        -r ${RESULTSPATH}/${dataset}/stats/${index}_fde${dfde}_k=${k}.csv \
        -k ${k} -d_fde ${dfde} -norm -no_rerank \
        -Ll 64 -Lr 2048 -Lmp 2
      echo ${index} ${dataset} ${dfde} ${k} done
    done
  done
elif [ "$index" == "muvera_norerank" ]; then
  for dfde in ${dfdes[@]}; do
    for k in ${ks[@]}; do
      bazel-bin/experiments/muvera_benchmark \
        -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
        -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
        -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
        -index ${RESULTSPATH}/${dataset}/indices/muvera_fde${dfde}.bin \
        -r ${RESULTSPATH}/${dataset}/stats/${index}_fde${dfde}_k=${k}.csv \
        -k ${k} -d_fde ${dfde} -no_rerank \
        -Ll 64 -Lr 2048 -Lmp 2
      echo ${index} ${dataset} ${dfde} ${k} done
    done
  done
elif [ "$index" == "muvera_normq" ]; then
  for dfde in ${dfdes[@]}; do
    for k in ${ks[@]}; do
      bazel-bin/experiments/muvera_benchmark \
        -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
        -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
        -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
        -index ${RESULTSPATH}/${dataset}/indices/muvera_fde${dfde}.bin \
        -r ${RESULTSPATH}/${dataset}/stats/${index}_fde${dfde}_k=${k}.csv \
        -k ${k} -d_fde ${dfde} -normq \
        -Ll 64 -Lr 2048 -Lmp 2
      echo ${index} ${dataset} ${dfde} ${k} done
    done
  done
elif [ "$index" == "muvera_norm_normq" ]; then
  for dfde in ${dfdes[@]}; do
    for k in ${ks[@]}; do
      bazel-bin/experiments/muvera_benchmark \
        -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
        -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
        -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
        -index ${RESULTSPATH}/${dataset}/indices/muvera_norm_fde${dfde}.bin \
        -r ${RESULTSPATH}/${dataset}/stats/${index}_fde${dfde}_k=${k}.csv \
        -k ${k} -d_fde ${dfde} -norm -normq \
        -Ll 64 -Lr 2048 -Lmp 2
      echo ${index} ${dataset} ${dfde} ${k} done
    done
  done
elif [ "$index" == "muvera_norm_norerank_normq" ]; then
  for dfde in ${dfdes[@]}; do
    for k in ${ks[@]}; do
      bazel-bin/experiments/muvera_benchmark \
        -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
        -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
        -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
        -index ${RESULTSPATH}/${dataset}/indices/muvera_norm_fde${dfde}.bin \
        -r ${RESULTSPATH}/${dataset}/stats/${index}_fde${dfde}_k=${k}.csv \
        -k ${k} -d_fde ${dfde} -norm -no_rerank -normq \
        -Ll 64 -Lr 2048 -Lmp 2
      echo ${index} ${dataset} ${dfde} ${k} done
    done
  done
elif [ "$index" == "muvera_norerank_normq" ]; then
  for dfde in ${dfdes[@]}; do
    for k in ${ks[@]}; do
      bazel-bin/experiments/muvera_benchmark \
        -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
        -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
        -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
        -index ${RESULTSPATH}/${dataset}/indices/muvera_fde${dfde}.bin \
        -r ${RESULTSPATH}/${dataset}/stats/${index}_fde${dfde}_k=${k}.csv \
        -k ${k} -d_fde ${dfde} -no_rerank -normq \
        -Ll 64 -Lr 2048 -Lmp 2
      echo ${index} ${dataset} ${dfde} ${k} done
    done
  done
elif [ "$index" == "mpv" ]; then
  for k in ${ks[@]}; do
    bazel-bin/experiments/mpv_benchmark \
      -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
      -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
      -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
      -index ${RESULTSPATH}/${dataset}/indices/mpv.bin \
      -r ${RESULTSPATH}/${dataset}/stats/${index}_fde${dfde}_k=${k}.csv \
      -k ${k} \
      -Ll 64 -Lr 2048 -Lmp 2
    echo ${index} ${dataset} ${k} done
  done
elif [ "$index" == "mpv_norm" ]; then
  for k in ${ks[@]}; do
    bazel-bin/experiments/mpv_benchmark \
      -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
      -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
      -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
      -index ${RESULTSPATH}/${dataset}/indices/mpv_norm.bin \
      -r ${RESULTSPATH}/${dataset}/stats/${index}_fde${dfde}_k=${k}.csv \
      -k ${k} -norm \
      -Ll 64 -Lr 2048 -Lmp 2
    echo ${index} ${dataset} ${k} done
  done
elif [ "$index" == "mpv_norm_norerank" ]; then
  for k in ${ks[@]}; do
    bazel-bin/experiments/mpv_benchmark \
      -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
      -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
      -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
      -index ${RESULTSPATH}/${dataset}/indices/mpv_norm.bin \
      -r ${RESULTSPATH}/${dataset}/stats/${index}_fde${dfde}_k=${k}.csv \
      -k ${k} -norm -no_rerank \
      -Ll 64 -Lr 2048 -Lmp 2
    echo ${index} ${dataset} ${k} done
  done
elif [ "$index" == "mpv_norerank" ]; then
  for k in ${ks[@]}; do
    bazel-bin/experiments/mpv_benchmark \
      -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
      -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
      -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
      -index ${RESULTSPATH}/${dataset}/indices/mpv.bin \
      -r ${RESULTSPATH}/${dataset}/stats/${index}_fde${dfde}_k=${k}.csv \
      -k ${k} -no_rerank \
      -Ll 64 -Lr 2048 -Lmp 2
    echo ${index} ${dataset} ${k} done
  done
elif [ "$index" == "mpv_normq" ]; then
  for k in ${ks[@]}; do
    bazel-bin/experiments/mpv_benchmark \
      -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
      -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
      -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
      -index ${RESULTSPATH}/${dataset}/indices/mpv.bin \
      -r ${RESULTSPATH}/${dataset}/stats/${index}_fde${dfde}_k=${k}.csv \
      -k ${k} -normq \
      -Ll 64 -Lr 2048 -Lmp 2
    echo ${index} ${dataset} ${k} done
  done
elif [ "$index" == "mpv_norm_normq" ]; then
  for k in ${ks[@]}; do
    bazel-bin/experiments/mpv_benchmark \
      -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
      -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
      -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
      -index ${RESULTSPATH}/${dataset}/indices/mpv_norm.bin \
      -r ${RESULTSPATH}/${dataset}/stats/${index}_fde${dfde}_k=${k}.csv \
      -k ${k} -norm -normq \
      -Ll 64 -Lr 2048 -Lmp 2
    echo ${index} ${dataset} ${k} done
  done
elif [ "$index" == "mpv_norm_norerank_normq" ]; then
  for k in ${ks[@]}; do
    bazel-bin/experiments/mpv_benchmark \
      -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
      -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
      -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
      -index ${RESULTSPATH}/${dataset}/indices/mpv_norm.bin \
      -r ${RESULTSPATH}/${dataset}/stats/${index}_fde${dfde}_k=${k}.csv \
      -k ${k} -norm -no_rerank -normq \
      -Ll 64 -Lr 2048 -Lmp 2
    echo ${index} ${dataset} ${k} done
  done
elif [ "$index" == "mpv_norerank_normq" ]; then
  for k in ${ks[@]}; do
    bazel-bin/experiments/mpv_benchmark \
      -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
      -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
      -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
      -index ${RESULTSPATH}/${dataset}/indices/mpv.bin \
      -r ${RESULTSPATH}/${dataset}/stats/${index}_fde${dfde}_k=${k}.csv \
      -k ${k} -no_rerank -normq \
      -Ll 64 -Lr 2048 -Lmp 2
    echo ${index} ${dataset} ${k} done
  done
fi