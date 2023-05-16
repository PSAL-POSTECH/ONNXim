#!/bin/bash
MODEL=$1
CONFIG=$2
MODEL_PATH=../models/$MODEL
CONFIG_PATH=../configs/timeloop_configs/$CONFIG

pids=""
for  LAYER in `ls $MODEL_PATH/*.yaml`; do
    srun -p allcpu ./timeloop_slurm_job.sh $MODEL $CONFIG $LAYER & 
    pids="$pids $!"
done

wait $pids

for MAP_FILE in `ls $MODEL_PATH/*.map`; do 
    echo $MAP_FILE
    MAP_FILE_BASE=`basename $MAP_FILE`
    ID="${MAP_FILE_BASE%.*}"
    MAPPING=`cat $MAP_FILE`
    echo $ID, $MAPPING >> $MODEL_PATH/$MODEL.mapping
    rm $MAP_FILE
done

echo "DONE"
