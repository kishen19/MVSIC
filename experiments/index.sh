index=$1
dataset=$2

DATAPATH=/ssd2/laxman/multivector
RESULTSPATH=/ssd2/kishen/MVC

mkdir -p logs
mkdir -p ${RESULTSPATH}/${dataset}
mkdir -p ${RESULTSPATH}/${dataset}/indices

maxsizes=(500)
iters=(5)
rounds=0

if [ "$index" == "mvivf" ]; then
  for maxsize in ${maxsizes[@]}; do
    for iter in ${iters[@]}; do
      bazel-bin/experiments/${index}_benchmark \
        -i ${DATAPATH}/${dataset}/${dataset}_points.pcs \
        -o ${RESULTSPATH}/${dataset}/indices/${index}_maxsize${maxsize}_iters${iter}.bin \
        -rounds ${rounds} -iters ${iter} -maxsize ${maxsize} -v > logs/${index}_${dataset}_maxsize${maxsize}_iters${iter}.log
      echo ${dataset} ${maxsize} ${iter} done
    done
  done
elif [ "$index" == "svh" ]; then
  for maxsize in ${maxsizes[@]}; do
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16 \
      bazel-bin/experiments/${index}_benchmark \
      -i ${DATAPATH}/${dataset}/${dataset}_points.pcs \
      -o ${RESULTSPATH}/${dataset}/indices/${index}_maxsize${maxsize}.bin \
      -rounds ${rounds} -iters ${iter} -maxsize ${maxsize} -v > logs/${index}_${dataset}_maxsize${maxsize}.log
    echo ${dataset} ${maxsize} ${iter} done
  done
fi