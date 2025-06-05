index=$1
dataset=$2

DATAPATH=/ssd2/laxman/multivector
RESULTSPATH=/ssd2/kishen/MVC

mkdir -p logs
mkdir -p ${RESULTSPATH}/${dataset}

maxsizes=(100 500)
iters=(10)
rounds=0

for maxsize in ${maxsizes[@]}; do
  for iter in ${iters[@]}; do
    subname=""
    if [ "$index" = "mvivf" ]; then
      subname="_iters${iter}"
    fi
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16 \
      bazel-bin/experiments/${index}_benchmark \
      -i ${DATAPATH}/${dataset}/${dataset}_points.pcs \
      -o ${RESULTSPATH}/${dataset}/${index}_maxsize${maxsize}${subname}.bin \
      -rounds ${rounds} -iters ${iter} -maxsize ${maxsize} -v > logs/${index}_${dataset}_maxsize${maxsize}${subname}.log
    echo ${dataset} ${maxsize} ${iter} done
  done
done