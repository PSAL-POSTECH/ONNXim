# ONNXim: A Fast, Cycle-level Multi-core NPU Simulator
[![Docker Image CI](https://github.com/PSAL-POSTECH/ONNXim/actions/workflows/docker-image.yml/badge.svg)](https://github.com/PSAL-POSTECH/ONNXim/actions/workflows/docker-image.yml)

ONNXim is a fast cycle-level simulator that can model multi-core NPUs for DNN inference. Its features include the following:
- Faster simulation speed in comparison to other _detailed_ NPU simulation frameworks (see the figure below).
- Support for modeling multi-core NPUs.
- Support for cycle-level simulation of memory (through [Ramulator](https://github.com/CMU-SAFARI/ramulator)) and network-on-chip (through [Booksim2](https://github.com/booksim/booksim2)), which is important for properly modeling memory-bound operations in deep learning.
- Use of ONNX graphs as DNN model specifications, enabling simulation of DNNs implemented in different deep learning frameworks (e.g., PyTorch and TensorFlow).
- Support language models that do not use ONNX graphs. Additionally, enable auto-regressive generation phases and iteration-level batching.

For more details, please refer to our [paper](https://ieeexplore.ieee.org/document/10726822)!

![Speedup](/img/speedup.png)
**Figure description**: we compare the simulation speed of ONNXim to that of [Accel-sim](https://accel-sim.github.io/) (a GPU simulator with Tensor Core model) as GPUs are widely used for deep learning and such a GPU simulator can be used to study systems for deep learning. We also include [SMAUG](https://github.com/harvard-acc/smaug) in the comparison. On the x-axis, we vary the size of each dimension for an NxNxN GEMM operation.

## Requirements
### OS Distribution
* ubuntu:20.04 (Recommended)

*We have not tested ONNXim on other Linux distributions.*
### Python(>=3.8) Packages
* torch >= 1.10.1
* conan == 1.57.0
* onnxruntime >= 1.10.0
* torchvision >= 0.17.2 (Optional: for ONNX graph generation)
* optimum >= 1.19.0 (Optional: for ONNX graph generation)

### Other Dependencies
* cmake >= 3.22.1
* gcc >= 10.5.0


## ONNX Graph
ONNXim requires ONNX graph files (.onnx) to simulate DNN models. We provide an example input file for fused ResNet-18 in the `models` directory. If you want to export a new DNN model as an ONNX Graph, you can use the `scripts/generate_*_onnx.py` scripts as shown below.

For ResNet-50:
```
$ cd ONNXim
$ python3 ./scripts/generate_cnn_onnx.py --model resnet50
```

For GPT and BERT:
```
$ cd ONNXim
$ python3 ./scripts/generate_transformer_onnx.py --model gpt2
$ python3 ./scripts/generate_transformer_onnx.py --model bert
```

## Custom format
ONNXim suppo
------------

## Hardware Configuration
`configs` directory contains example NPU configration files in the JSON format.

```
  "num_cores" : 4,              // Number of NPU cores
  "core_type" : "systolic_ws",  // Core's data flow (Only weight stationary is supported)
  "core_freq" : 1000,           // Core's frequency (MHz)
  "core_width" : 128,           // Systolic array width
  "core_height" : 128,          // Systolic array height

  "spad_size" : 65536,          // Scratchpad size (KB)
  "accum_spad_size" : 8192,     // Accumulator SRAM size (KB)
  "sram_width" : 32,            // SRAM word size (B)

  "vector_process_bit" : 65536, // Vector unit compute throughput (bit)
  "add_latency" : 1,            // Vector add latency (cycle)
  "mul_latency" : 1,            // Vector mul latency (cycle)
  "exp_latency" : 1,            // Vector exp latency (cycle)
  "gelu_latency" : 1,           // Vector gelu latency (cycle)
  "add_tree_latency" : 1,       // Adder tree latency (cycle)
  "scalar_sqrt_latency" : 1,    // Scalar square root latency (cycle)
  "scalar_add_latency" : 1,     // Scalar add latency (cycle)
  "scalar_mul_latency" : 1,     // Scalar mul latency (cycle)

  "dram_type" : "ramulator",    // DRAM type (ex. ramulator, simple)
  "dram_freq" : 877,            // DRAM frequency (MHz)
  "dram_channels": 32,          // Number of DRAM channels
  "dram_req_size": 32,          // DRAM request size (B)
  "dram_latency" : 10,          // DRAM latency (cycle)
  "dram_print_interval": 10000, // DRAM stat print interval (cycle)
  "dram_config_path" : "../configs/ramulator_configs/HBM-config.cfg", // Ramulator config file path

  "icnt_type" : "simple",       // Interconnect type (ex. booksim, simple)
  "icnt_latency" : 1,           // Interconnect latency (cycle)
  "icnt_freq" : 2000,           // Interconnect frequency (MHz)
  "icnt_config_path" : "../configs/booksim2_configs/fly_c4_m32.icnt", // Booksim2 config file path

  "precision" : 2,              // Element's precision in tensor (Byte)
  "layout" : "NHWC",            // Data Layout
  "scheduler" : "simple"        // Scheduler type (ex. simple, spatial_split, time_multiplex, partition_cpu)
```
------------

# Getting Started
This section describes how to build and run ONNXim with a container-based method and a manual build method.
## 1. Container-based Method using Docker (Recommended)
Build a Docker image using the provided Dockerfile.
```
$ git clone https://github.com/PSAL-POSTECH/ONNXim.git 
$ cd ONNXim
$ docker build . -t onnxim
```

Run the docker image and the simulator.
```
$ docker run -it onnxim
(docker) cd /workspace/ONNXim
(docker) ./build/bin/Simulator --config ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4.json --model ./example/models_list.json
```


## 2. Manual Method
### Installation
```
$ git clone https://github.com/PSAL-POSTECH/ONNXim.git
$ cd ONNXim
$ git submodule update --recursive --init
```
### Build
```
$ mkdir build && cd build
$ conan install ..
$ cmake ..
$ make -j
```
### Run Simulator
```
$ cd ..
$ ./build/bin/Simulator --config ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4.json --model ./example/models_list.json
```

ONNXim supports custom model formats, with models like Llama and OPT implemented using this feature. Based on this, iteration-level scheduling policy is implemented.

Below is an example of how to execute it (**Note**: You have to add `--language` option):

```
$ ./build/bin/Simulator --config ./configs/systolic_ws_128x128_c4_simple_noc_tpuv4.json --models_list example/language_models.json --mode language
```

`language_models.json` is structured as follows:
```
{
  "models": [
    {
      "name": "opt-125m",
      "trace_file": "input.csv",
      "scheduler": "simple",
      "scheduler_config": {
        "max_batch_size": 8
      }
    }
  ]
}
```
- name: Specifies the LLM model to be selected.
- trace_file: Sets the request trace file.
- scheduler: Defines the scheduling policy to be used.

------------
## Result

![Demo](/img/ONNXim_demo.png)

------------
## Mapping
ONNXim uses a hierarchical tiling method that can handle large tensors. 
If the mapping method is not specified, the tiling method from [Gemmini](https://github.com/ucb-bar/gemmini) [DAC'21] is used by default.

### Manual Mapping File (Optional)
You can specify the mapping by placing a `*.mapping` file in the same folder as the `*.onnx` file.

The mapping file is composed of three parts:

1. Total Loop (e.g., `[T] N1 C3 M64 P112 Q112 S7 R7`)
2. Outer Loop (e.g., `[O] N1 C1 M4 P5 Q6 S1 R1`)
3. Inner Loop (e.g., `[I] N1 C3 M16 P23 Q22 S7 R7`)

where `N` stands for Batch Size, `C` for Input Channel, `M` for Output Channel, `P` for Output Rows, `Q` for Output Columns, `S` for Kernel Rows, `R` for Kernel Columns.

The `Total Loop` provides the overall loop information for the given layer. In the example above, `Total Loop` corresponds to a convolution operation with an output dimension of (N:1, M:64, P:112, Q:112) and a kernel dimension of (C:3, S:7, R:7, M:64).

The `Outer Loop` specifies how many times the `Inner Loop` needs to be iterated. In this example, the `Total Loop` has `P`=112 and the `Inner Loop` has `P`=23. Therefore, the `Outer Loop` should have `P`=ceiling(112/23)=5.

The `Inner Loop` determines the sizes of the input and weight tiles loaded to the scratchpad memory and the size of the output tile mapped to the accumulator.

In this example, assuming a 4-byte (i.e., FP32) data format, the size of the output tile will be 4x16x23x22=32384 bytes. The weight tile size will be 4x16x3x7x7=9408 bytes and the size of the (padded) input tile will be 4x1x3x29x28=9744 bytes. 

**Note**: The size of the input and weight tiles should not exceed half the size of the scratchpad memory for double buffering. Similarly, the size of the output tile should not exceed half the size of the accumulator. 

Below is an example mapping for ResNet-18.

```
[T] N1 C3 M64 P112 Q112 S7 R7 - [O] N1 C1 M4 P5 Q6 S1 R1 - [I] N1 C3 M16 P23 Q22 S7 R7
[T] N1 C64 M64 P56 Q56 S3 R3 - [O] N1 C1 M4 P3 Q3 S1 R1 - [I] N1 C64 M16 P23 Q22 S3 R3
[T] N1 C64 M64 P56 Q56 S3 R3 - [O] N1 C1 M4 P3 Q3 S1 R1 - [I] N1 C64 M16 P23 Q22 S3 R3
[T] N1 C64 M128 P28 Q28 S3 R3 - [O] N1 C2 M8 P2 Q2 S1 R1 - [I] N1 C51 M16 P23 Q22 S3 R3
[T] N1 C64 M128 P28 Q28 S1 R1 - [O] N1 C1 M8 P2 Q2 S1 R1 - [I] N1 C64 M16 P23 Q22 S1 R1
[T] N1 C128 M128 P28 Q28 S3 R3 - [O] N1 C1 M8 P2 Q2 S1 R1 - [I] N1 C128 M16 P23 Q22 S3 R3
[T] N1 C128 M256 P14 Q14 S3 R3 - [O] N1 C2 M7 P1 Q1 S1 R1 - [I] N1 C104 M40 P14 Q14 S3 R3
[T] N1 C256 M256 P14 Q14 S3 R3 - [O] N1 C2 M7 P1 Q1 S1 R1 - [I] N1 C210 M40 P14 Q14 S3 R3
[T] N1 C128 M256 P14 Q14 S1 R1 - [O] N1 C1 M7 P1 Q1 S1 R1 - [I] N1 C128 M40 P14 Q14 S1 R1
[T] N1 C128 M256 P14 Q14 S1 R1 - [O] N1 C1 M7 P1 Q1 S1 R1 - [I] N1 C128 M40 P14 Q14 S1 R1
[T] N1 C256 M512 P7 Q7 S3 R3 - [O] N1 C3 M5 P1 Q1 S1 R1 - [I] N1 C109 M104 P7 Q7 S3 R3
[T] N1 C256 M512 P7 Q7 S1 R1 - [O] N1 C1 M4 P1 Q1 S1 R1 - [I] N1 C256 M160 P7 Q7 S1 R1
[T] N1 C512 M512 P7 Q7 S3 R3 - [O] N1 C5 M5 P1 Q1 S1 R1 - [I] N1 C120 M112 P7 Q7 S3 R3
[T] N1 C512 M1000 - [O] N1 C1 M5 - [I] N1 C512 M248
```

------------
## Future Work
This current version only supports GEMM, Conv, Attention, GeLU, LayerNorm operations. Other operations will be supported in later versions.

## Citation
If you use ONNXim for your research, please cite the following paper.
```
@ARTICLE{10726822,
  author={Ham, Hyungkyu and Yang, Wonhyuk and Shin, Yunseon and Woo, Okkyun and Heo, Guseul and Lee, Sangyeop and Park, Jongse and Kim, Gwangsun},
  journal={IEEE Computer Architecture Letters}, 
  title={ONNXim: A Fast, Cycle-Level Multi-Core NPU Simulator}, 
  year={2024},
  volume={23},
  number={2},
  pages={219-222},
  keywords={Random access memory;Computational modeling;Vectors;Kernel;Tensors;Runtime;Libraries;Deep learning;Artificial neural networks;Systolic arrays;DNN inference;multi-tenancy;NPU;ONNX;simulator},
  doi={10.1109/LCA.2024.3484648}}
```
