#!/bin/bash

#models=("gpt2_s" "gpt2_g" "bert")
models=("multi_4_0_0_0_200_1111_1000" "multi_4_0_0_0_200_8111_1000" "multi_4_0_0_0_200_31111_1000")
batch_list=("1")
i=1
configs=("systolic_ws_128x128_c4_simple_noc_tpuv4_partition_quad") # "systolic_ws_128x128_c4_booksim2_tpuv4")
no_batch=1

for model_file in "${models[@]}"; do
    for batch in "${batch_list[@]}"; do
        for config in "${configs[@]}"; do
            for (( j=0; j<i; j++ )); do
                if [ "$no_batch" -eq 0 ]; then
                    echo "$ONNXIM_HOME/scripts/run_sbatch_docker.sh $config "$model_file"_$batch $batch"
                    $ONNXIM_HOME/scripts/run_sbatch_docker.sh $config "$model_file"_$batch $batch
                else
                    echo "$ONNXIM_HOME/scripts/run_sbatch_docker.sh $config "$model_file"_$batch $batch"
                    $ONNXIM_HOME/scripts/run_sbatch_docker.sh $config "$model_file" $batch
                fi
            done
        done
    done
done