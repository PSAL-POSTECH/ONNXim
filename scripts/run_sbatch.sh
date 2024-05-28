#!/bin/bash

models=("gpt2_s" "gpt2_g" "bert")
batch_list=("1" "2" "4" "8" "16" "32")
i=3
configs=("systolic_ws_128x128_c4_simple_noc_tpuv4") # "systolic_ws_128x128_c4_booksim2_tpuv4")
no_batch=1

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
    fi
    for batch in "${batch_list[@]}"; do
        if [ ! -d "$ONNXIM_HOME/results/$model_file/$batch" ]; then
            mkdir $ONNXIM_HOME/results/$model_file/$batch
        fi
        for config in "${configs[@]}"; do
            if [ ! -d "$ONNXIM_HOME/results/$model_file/$batch/$config" ]; then
                mkdir $ONNXIM_HOME/results/$model_file/$batch/$config
            fi
            for (( j=0; j<i; j++ )); do
                if [ "$no_batch" -eq 0 ]; then
                    echo "sbatch -J ONNXIM-$model_file -o $ONNXIM_HOME/results/$model_file/$batch/$config/result_$j.out  -e $ONNXIM_HOME/results/$model_file/$batch/$config/result_$j.err $ONNXIM_HOME/scripts/onnxim_sbatch.sh $config "$model_file"_$batch"
                    sbatch -J ONNXIM-$model_file -o $ONNXIM_HOME/results/$model_file/$batch/$config/result_$j.out  -e $ONNXIM_HOME/results/$model_file/$batch/$config/result_$j.err $ONNXIM_HOME/scripts/onnxim_sbatch.sh $config "$model_file"_$batch
                else
                    echo "sbatch -J ONNXIM-$model_file -o $ONNXIM_HOME/results/$model_file/$batch/$config/result_$j.out  -e $ONNXIM_HOME/results/$model_file/$batch/$config/result_$j.err $ONNXIM_HOME/scripts/onnxim_sbatch.sh $config "$model_file""
                    sbatch -J ONNXIM-$model_file -o $ONNXIM_HOME/results/$model_file/$batch/$config/result_$j.out  -e $ONNXIM_HOME/results/$model_file/$batch/$config/result_$j.err $ONNXIM_HOME/scripts/onnxim_sbatch.sh $config "$model_file"
                fi
            done
        done
    done
done