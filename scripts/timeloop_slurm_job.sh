#!/bin/bash
MODEL=$1
CONFIG=$2
LAYER=$3
MODEL_PATH=../models/$MODEL
CONFIG_PATH=../configs/timeloop_configs/$CONFIG

execute_timeloop() {
    echo $LAYER
    LAYER_FILE=`basename $LAYER`
    ID="${LAYER_FILE%.*}"
    TMP_DIR=tmp-$ID
    echo $TMP_DIR
    mkdir $TMP_DIR
    pushd $TMP_DIR
        ../timeloop-mapper ../$CONFIG_PATH/arch/*.yaml ../$CONFIG_PATH/arch/components/*.yaml ../$CONFIG_PATH/mapper/mapper.yaml ../$CONFIG_PATH/constraints/*.yaml ../$LAYER > /dev/null 2>/dev/null
        mv map.tmp.txt ../$MODEL_PATH/$ID.map
    popd
    rm -rf $TMP_DIR
}

execute_timeloop
