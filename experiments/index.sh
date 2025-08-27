index=$1
dataset=$2

DATAPATH=/ssd2/laxman/multivector
RESULTSPATH=/ssd2/kishen/MVC

mkdir -p logs
mkdir -p ${RESULTSPATH}/${dataset}
mkdir -p ${RESULTSPATH}/${dataset}/indices

maxsizes=(500)
iters=(5)
Rs=(128 256 512)
dfdes=(2560 5120 10240 20480)

rounds=0

if [ "$index" == "mvivf" ]; then
  for maxsize in ${maxsizes[@]}; do
    for iter in ${iters[@]}; do
      bazel-bin/experiments/${index}_benchmark \
        -i ${DATAPATH}/${dataset}/${dataset}_points.pcs \
        -o ${RESULTSPATH}/${dataset}/indices/${index}_maxsize${maxsize}_iters${iter}.bin \
        -rounds ${rounds} -iters ${iter} -maxsize ${maxsize} > logs/${index}_${dataset}_maxsize${maxsize}_iters${iter}.log
      echo ${index} ${dataset} ${maxsize} ${iter} done
    done
  done
elif [ "$index" == "svh" ]; then
  for maxsize in ${maxsizes[@]}; do
    bazel-bin/experiments/${index}_benchmark \
      -i ${DATAPATH}/${dataset}/${dataset}_points.pcs \
      -o ${RESULTSPATH}/${dataset}/indices/${index}_maxsize${maxsize}.bin \
      -rounds ${rounds} -maxsize ${maxsize} > logs/${index}_${dataset}_maxsize${maxsize}.log
    echo ${index} ${dataset} ${maxsize} done
  done
elif [ "$index" == "vamana" ]; then
  for R in ${Rs[@]}; do
    bazel-bin/experiments/${index}_benchmark \
      -i ${DATAPATH}/${dataset}/${dataset}_points.pcs \
      -o ${RESULTSPATH}/${dataset}/indices/${index}_R${R}.bin \
      -rounds ${rounds} -R ${R} -v > logs/${index}_${dataset}_R${R}.log
    echo ${index} ${dataset} ${R} done
  done
elif [ "$index" == "muvera" ]; then
  for dfde in ${dfdes[@]}; do
    bazel-bin/experiments/${index}_benchmark \
      -i ${DATAPATH}/${dataset}/${dataset}_points.pcs \
      -o ${RESULTSPATH}/${dataset}/indices/${index}_fde${dfde}.bin \
      -rounds ${rounds} -d_fde ${dfde} -v > logs/${index}_${dataset}_fde${dfde}.log
    echo ${index} ${dataset} ${dfde} done
  done
elif [ "$index" == "mpv" ]; then
  bazel-bin/experiments/${index}_benchmark \
    -i ${DATAPATH}/${dataset}/${dataset}_points.pcs \
    -o ${RESULTSPATH}/${dataset}/indices/${index}.bin \
    -rounds ${rounds} -v > logs/${index}_${dataset}.log
  echo ${index} ${dataset} done
fi