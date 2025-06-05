index=$1
dataset=$2
reduced=""
# reduced="_reduced"

DATAPATH=/ssd2/laxman/multivector
RESULTSPATH=/ssd2/kishen/MVC

maxsizes=(100 500)
iters=(10)
ks=(10 100)
cands=(1 4 16 64)

if [ "$index" == "mvivf" ]; then
  for maxsize in ${maxsizes[@]}; do
    for iter in ${iters[@]}; do
      for k in ${ks[@]}; do
        OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16 \
        bazel-bin/experiments/${index}_benchmark \
          -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
          -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
          -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
          -index ${RESULTSPATH}/${dataset}/${index}_maxsize${maxsize}_iters${iter}.bin \
          -r ${RESULTSPATH}/${dataset}/${index}_maxsize${maxsize}_iters${iter}_k=${k}.csv \
          -k ${k} \
          -npl 1 -npr 128 -npmp 2 -npad 0
        echo ${dataset} ${maxsize} ${iter} ${k} done
      done
    done
  done
elif [ "$index" == "svh" ]; then
  for maxsize in ${maxsizes[@]}; do
    for cand in ${cands[@]}; do
      for k in ${ks[@]}; do
        prod=$((cand*k))
        OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16 \
        bazel-bin/experiments/${index}_benchmark \
          -i  ${DATAPATH}/${dataset}/${dataset}_points.pcs  \
          -q  ${DATAPATH}/${dataset}/${dataset}${reduced}_queries.pcs \
          -gt ${DATAPATH}/${dataset}/${dataset}${reduced}_chamfer_neighbors.gt  \
          -index ${RESULTSPATH}/${dataset}/${index}_maxsize${maxsize}.bin \
          -r ${RESULTSPATH}/${dataset}/${index}_maxsize${maxsize}_k=${k}_cand=${cand}.csv \
          -k ${k} -cands ${prod} \
          -npl 1 -npr 128 -npmp 2 -npad 0 
        echo ${dataset} ${maxsize} ${cand} ${k} done
      done
    done
  done
fi