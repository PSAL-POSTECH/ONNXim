#!/bin/bash
#models=("matmul_1024_1024_1024" "matmul_2048_2048_2048" "matmul_4096_4096_4096" "matmul_8192_8192_8192")
#models=("matmul_2048") #"matmul_32" "matmul_64" "matmul_128" "matmul_256" "matmul_512" "matmul_1024" "matmul_2048" "conv_64" "conv_256" "conv_1024")
#models=("matmul_32_32_32" "matmul_64_64_64" "matmul_128_128_128" "matmul_256_256_256" "matmul_512_512_512")
models=("multi_1_0_2_0_100_1121_1000_once" "multi_2_0_2_0_100_8121_1000_once" "multi_8_0_2_0_100_32121_1000_once")
#models=("matmul_512_512_1024" "matmul_512_1024_2" "matmul_512_1024_512" "matmul_512_1024_3072" "matmul_512_1024_4096" "matmul_512_4096_1024")
#models=("conv_64" "conv_256" "conv_1024")
#configs=("systolic_ws_8x8_c1_simple_noc_transformer") # "systolic_ws_8x8_c1_booksim2_transformer" "systolic_ws_8x8_c4_simple_noc_transformer" "systolic_ws_8x8_c4_booksim2_transformer")
#configs=("systolic_ws_8x8_c4_simple_noc_transformer" "systolic_ws_8x8_c4_booksim2_transformer")
configs=("systolic_ws_128x128_c4_simple_noc_tpuv4_partition_quad") #"systolic_ws_128x128_c4_booksim2_tpuv4")
#models=("matmul_4096_4096_4096" "matmul_8192_8192_8192") #("matmul_1024_1024_1024" "matmul_2048_2048_2048" "matmul_4096_4096_4096" "matmul_8192_8192_8192")
i=5

#python3 $ONNXIM_HOME/scripts/generate_matmul_onnx.py
#python3 $ONNXIM_HOME/scripts/generate_conv_onnx.py

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
            $ONNXIM_HOME/build/bin/Simulator --config ./configs/$config.json --model $ONNXIM_HOME/model_lists/$model_file.json > $ONNXIM_HOME/results/$model_file/$config/result_$j &
            simulation_time=$(grep "Simulation time:" "$ONNXIM_HOME/results/$model_file/$config/result_$j" | awk '{print $(NF-1)}')
            if [[ ! -z "$simulation_time" ]]; then
                total_time=$(echo "$total_time + $simulation_time" | bc)
            fi
        done
        mean_time=$(awk "BEGIN {print $total_time / $i}")
        echo "Mean Simulation time: $mean_time seconds"
    done
done
wait