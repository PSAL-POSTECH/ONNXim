#!/bin/bash

#SBATCH -p allcpu
#SBATCH --nodes=1
#SBATCH --nodelist=n10
#SBATCH --ntasks-per-node=1

ml swap gnu8 gnu13
which gcc

echo "config: $1 model: $2"
echo "$ONNXIM_HOME/build/bin/Simulator --config $ONNXIM_HOME/configs/$1.json --model $ONNXIM_HOME/model_lists/$2.json"
$ONNXIM_HOME/build/bin/Simulator --config $ONNXIM_HOME/configs/$1.json --model $ONNXIM_HOME/model_lists/$2.json

exit 0