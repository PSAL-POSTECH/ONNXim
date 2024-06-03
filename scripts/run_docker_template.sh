#!/bin/bash
cd /home/workspace
export ONNXIM_HOME=/home/workspace
CONFIG_FILE=$1
MODEL_FILE=$2

echo "Running experiment with config file: $CONFIG_FILE, model file: $MODEL_FILE"
$ONNXIM_HOME/build/bin/Simulator --config $ONNXIM_HOME/configs/$1.json \
    --model $ONNXIM_HOME/model_lists/$2.json