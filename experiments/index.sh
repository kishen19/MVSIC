dataset=$1

DATAPATH=/ssd2/laxman/multivector
RESULTSPATH=/ssd2/kishen/MVC

mkdir -p logs
mkdir -p ${RESULTSPATH}/${dataset}

maxsizes=(500)
iters=(5)
rounds=0

for maxsize in ${maxsizes[@]}; do
  for iter in ${iters[@]}; do
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16 \
      bazel-bin/experiments/mvivf_benchmark \
      -i ${DATAPATH}/${dataset}/${dataset}_points.pcs \
      -o ${RESULTSPATH}/${dataset}/index_maxsize${maxsize}_iters${iter}.bin \
      -rounds ${rounds} -iters ${iter} -maxsize ${maxsize} > logs/${dataset}_maxsize${maxsize}_iters${iter}.log
    echo ${dataset} ${maxsize} ${iter} done
  done
done