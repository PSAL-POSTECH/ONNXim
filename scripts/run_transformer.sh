#!/bin/bash

models=("gpt2_s" "gpt2_g" "bert")
batch_list=("1" "2" "4" "8" "16" "32")

configs=("systolic_ws_128x128_c4_simple_noc_tpuv4" "systolic_ws_128x128_c4_booksim2_tpuv4")
i=1

#python3.8 $ONNXIM_HOME/scripts/generate_transformer_onnx.py --model gpt2
#python3 $ONNXIM_HOME/scripts/generate_transformer_onnx.py --model bert

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
            total_time=0
            for (( j=0; j<i; j++ )); do
                echo "$ONNXIM_HOME/build/bin/Simulator --config $ONNXIM_HOME/configs/$config.json --model $ONNXIM_HOME/model_lists/"$model_file"_$batch.json > $ONNXIM_HOME/results/$model_file/$batch/$config/result_$j 2>&1"
                $ONNXIM_HOME/build/bin/Simulator --config $ONNXIM_HOME/configs/$config.json --model $ONNXIM_HOME/model_lists/"$model_file"_$batch.json > $ONNXIM_HOME/results/$model_file/$batch/$config/result_$j 2>&1
                simulation_time=$(grep "Simulation time:" "$ONNXIM_HOME/results/$model_file/$batch/$config/result_$j" | awk '{print $(NF-1)}')
                if [[ ! -z "$simulation_time" ]]; then
                    total_time=$(echo "$total_time + $simulation_time" | bc)
                fi
            done
            mean_time=$(awk "BEGIN {print $total_time / $i}")
            echo "Mean Simulation time: $mean_time seconds"
        done
    done
done