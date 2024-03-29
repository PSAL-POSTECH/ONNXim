#!/bin/bash

models=("resnet50_vgg16")
i=1
configs=("systolic_ws_128x128_c4_simple_noc_tpuv4") # "systolic_ws_128x128_c4_booksim2_tpuv4")

if [ ! -d "$ONNXIM_HOME/results" ]; then
    mkdir $ONNXIM_HOME/results
fi

for model_file in "${models[@]}"; do
    if [ ! -d "$ONNXIM_HOME/results/$model_file" ]; then
        mkdir $ONNXIM_HOME/results/$model_file
    fi
    if [[ $model_file == "gpt2_g" ]] || [[ $model_file == "gpt2_s" ]]; then
        onnx_file="gpt2"
    elif [[ $model_file == "bert" ]]; then
        onnx_file="$model_file"
    else
        onnx_file="$model_file"
    fi
    if [ ! -d "$ONNXIM_HOME/results/$model_file" ]; then
        mkdir $ONNXIM_HOME/results/$model_file
    fi
    for config in "${configs[@]}"; do
        if [ ! -d "$ONNXIM_HOME/results/$model_file/$config" ]; then
            mkdir $ONNXIM_HOME/results/$model_file/$config
        fi
        for (( j=0; j<i; j++ )); do
            echo "sbatch -J ONNXIM-$model_file -o $ONNXIM_HOME/results/$model_file/$config/result_$j.out  -e $ONNXIM_HOME/results/$model_file/$config/result_$j.err $ONNXIM_HOME/scripts/onnxim_sbatch.sh $config "$model_file""
            sbatch -J ONNXIM-$model_file -o $ONNXIM_HOME/results/$model_file/$config/result_$j.out  -e $ONNXIM_HOME/results/$model_file/$config/result_$j.err $ONNXIM_HOME/scripts/onnxim_sbatch.sh $config "$model_file"
        done
    done
done