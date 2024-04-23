# ONNXim: Fast and Detailed Multi-core NPU Simulator
[![Docker Image CI](https://github.com/PSAL-POSTECH/ONNXim/actions/workflows/docker-image.yml/badge.svg)](https://github.com/PSAL-POSTECH/ONNXim/actions/workflows/docker-image.yml)

ONNXim is a fast and detailed multi-core NPU
simulator. 
- ONNXim provide fastsimulation speed. 
- ONNXim supports multi-core simulation and detailed modeling of DRAM and interconnect, enabling the modeling of contention when running multiple DNN models simultaneously.
- ONNXim is compatible with ONNX graphs, allowing simulation of DNN models across multiple frameworks without
requiring code modifications

## Requirements
### OS Distribution
* Centos8 (Recommended)

*Other OS distributions are not tested!*
### Python(>=3.8) packages
* torch >= 1.10.1
* conan == 1.57.0
* onnxruntime >= 1.10.0
* torchvision >= 0.17.2 (Optional: for onnx graph generation)
* optimum >= 1.19.0 (Optional: for onnx graph generation)

### Package
* cmake >= 3.22.1 (You may need to build manually)
* gcc >= 8.3


## ONNX graph
ONNXim require ONNX graph file (.onnx) to simulate DNN model. We provide fused ResNet18 in `models` directory already. If you want to export new DNN model to ONNX grpah, you can use `scripts/generate_*_onnx.py`.

If you want ResNet50 model, follow the example below
```
$ cd ONNXim
$ python3 ./srcripts/generate_cnn_onnx.py --model resnet50
```

In case of GPT2 or BERT,
```
$ cd ONNXim
$ python3 ./scripts/generate_transformer_onnx.py --model gpt2
$ python3 ./scripts/generate_transformer_onnx.py --model bert
```

------------

## Hardware Configuration
`configs` directory contains several JSON files that store information about hardware configurations.

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
This section describes how to build and run ONNXim. There are two methods to run ONNXim: Container-based method and Manual build method.
## 1. Docker image method (Recommended)
```
$ git clone https://github.com/PSAL-POSTECH/ONNXim.git 
$ cd ONNXim
$ docker build . -t onnxim
```
Build a container image using the provided Dockerfile.

```
$ docker run -it onnxim
(docker) cd ONNXim
(docker) ./build/bin/Simulator --config ./configs/systolic_ws_128x128_c4_simple_noc.json --model ./models_list.json
```
Run docker image and simulate resnet18 example


## 2. Manual method
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
$ ./build/bin/Simulator --config ./configs/systolic_ws_8x8_c1_simple_noc.json --model ./models_list.json
```

------------
## Result

![Demo](/img/ONNXim_demo.png)

------------
## Mapping (Optional)
ONNXim uses a hierarchical tiling method that can handle large tensor. 
If the mapping method is not specified, the tiling method of Gemmini is used by default.

### Manual Mapping file 
You can specify the mapping method by putting `*.mapping` file in the same folder of `*.onnx` file.

Mapping file is composed to 3 parts.

1. Total Loop: `[T] N1 C3 M64 P112 Q112 S7 R7`
2. Outer Loop: `[O] N1 C1 M4 P5 Q6 S1 R1`
3. Inner Loop: `[I] N1 C3 M16 P23 Q22 S7 R7`

N: Batch size, C: Input channel, M: Output Channel, P: Output Rows, Q: Output Cols, S: Kernel Row, R: Kernel Cols

This mapping is an example of first convolution layer in ResNet18. Inner Loop is the tensor size that can be holded in scratch pad and accumulator.

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
Activation and Weight are stored in scratch pad, and output is stored in accumulator. This simulator consider the size of scratch pad as 256KB and accumulator size as 16KB (default, you can modify). Furthermore, it uses double buffering so that it could fill the scratch pad with half size. Mapping is calculated before implement simulator.

------------
## Future Works
This version only supports GEMM, Conv, Attention, GeLU, LayerNorm Operation. We're developing the simulator to support other operations such as pooling.
