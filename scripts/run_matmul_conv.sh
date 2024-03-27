#!/bin/bash

models=("matmul_2048") #"matmul_32" "matmul_64" "matmul_128" "matmul_256" "matmul_512" "matmul_1024" "matmul_2048" "conv_64" "conv_256" "conv_1024")
#models=("conv_64" "conv_256" "conv_1024")
configs=("systolic_ws_8x8_c1_simple_noc_transformer" "systolic_ws_8x8_c1_booksim2_transformer" "systolic_ws_8x8_c4_simple_noc_transformer" "systolic_ws_8x8_c4_booksim2_transformer")
i=5

python3 $ONNXIM_HOME/scripts/generate_matmul_onnx.py
python3 $ONNXIM_HOME/scripts/generate_conv_onnx.py

if [ ! -d "$ONNXIM_HOME/results" ]; then
    mkdir $ONNXIM_HOME/results
fi
 
for model_file in "${models[@]}"; do
    if [ ! -d "$ONNXIM_HOME/results/$model_file" ]; then
        mkdir $ONNXIM_HOME/results/$model_file
    fi
    for config in "${configs[@]}"; do
        if [ ! -d "$ONNXIM_HOME/results/$model_file/$config" ]; then
            mkdir $ONNXIM_HOME/results/$model_file/$config
        fi
        total_time=0
        for (( j=0; j<i; j++ )); do
            echo "$ONNXIM_HOME/build/bin/Simulator --config $ONNXIM_HOME/configs/$config.json --model $ONNXIM_HOME/model_lists/$model_file.json > $ONNXIM_HOME/results/$model_file/$config/result_$j"
            $ONNXIM_HOME/build/bin/Simulator --config ./configs/$config.json --model $ONNXIM_HOME/model_lists/$model_file.json > $ONNXIM_HOME/results/$model_file/$config/result_$j
            simulation_time=$(grep "Simulation time:" "$ONNXIM_HOME/results/$model_file/$config/result_$j" | awk '{print $(NF-1)}')
            if [[ ! -z "$simulation_time" ]]; then
                total_time=$(echo "$total_time + $simulation_time" | bc)
            fi
        done
        mean_time=$(awk "BEGIN {print $total_time / $i}")
        echo "Mean Simulation time: $mean_time seconds"
    done
done