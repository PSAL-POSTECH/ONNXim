# ONNXim: Fast and Detailed Multi-core NPU Simulator
[![Docker Image CI](https://github.com/PSAL-POSTECH/ONNXim/actions/workflows/docker-image.yml/badge.svg)](https://github.com/PSAL-POSTECH/ONNXim/actions/workflows/docker-image.yml)

ONNXim is a fast cycle-level simulator that models multi-core NPUs for DNN inference. Its features include the following:
- Fast simulation speed. The simulation speed comparison below shows the result that ONNXim is 30 to 40 times faster than Accel-Sim.
- Support for modeling multi-core NPUs.
- Support for cycle-level simulation of network-on-chip (through [Booksim2](https://github.com/booksim/booksim2)) and memory (through [Ramulator](https://github.com/CMU-SAFARI/ramulator)), which is important for memory-bound operations of DNNs.
- Use of ONNX graphs as DNN model specifications, enabling simulation of DNNs implemented in different deep learning frameworks (e.g., PyTorch and TensorFlow).

![Speedup](/img/speedup.png)
## Requirements
### OS Distribution
* CentOS 8 (Recommended)

*We have not tested ONNXim on other Linux distributions.*
### Python(>=3.8) Packages
* torch >= 1.10.1
* conan == 1.57.0
* onnxruntime >= 1.10.0
* torchvision >= 0.17.2 (Optional: for ONNX graph generation)
* optimum >= 1.19.0 (Optional: for ONNX graph generation)

### Other Dependencies
* cmake >= 3.22.1
* gcc >= 8.3


## ONNX Graph
ONNXim requires ONNX graph files (.onnx) to simulate DNN models. We provide a fused ResNet-18 in `models` directory as an example. If you want to export a new DNN model to an ONNX Graph, you can use the `scripts/generate_*_onnx.py` script as shown below.

For ResNet-50:
```
$ cd ONNXim
$ python3 ./srcripts/generate_cnn_onnx.py --model resnet50
```

For GPT and BERT:
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
## 1. Docker Image Method (Recommended)
Build a container image using the provided Dockerfile.
```
$ git clone https://github.com/PSAL-POSTECH/ONNXim.git 
$ cd ONNXim
$ docker build . -t onnxim
```

Run docker image and simulate ResNet-18 example
```
$ docker run -it onnxim
(docker) cd ONNXim
(docker) ./build/bin/Simulator --config ./configs/systolic_ws_128x128_c4_simple_noc.json --model ./models_list.json
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
$ ./build/bin/Simulator --config ./configs/systolic_ws_128x128_c4_simple_noc.json --model ./models_list.json
```

------------
## Result

![Demo](/img/ONNXim_demo.png)

------------
## Mapping
ONNXim uses a hierarchical tiling method that can handle large tensors. 
If the mapping method is not specified, the tiling method from [Gemmini](https://github.com/ucb-bar/gemmini) [DAC'21] is used by default.

### Manual Mapping file (Optional)
You can specify the mapping by putting `*.mapping` file in the same folder as a `*.onnx` file.

The mapping file is composed of 3 parts.

1. Total Loop: `[T] N1 C3 M64 P112 Q112 S7 R7`
2. Outer Loop: `[O] N1 C1 M4 P5 Q6 S1 R1`
3. Inner Loop: `[I] N1 C3 M16 P23 Q22 S7 R7`

`N`: Batch Size, `C`: Input Channel, `M`: Output Channel, `P`: Output Rows, `Q`: Output Cols, `S`: Kernel Row, `R`: Kernel Cols

The `Total Loop` stores the loop information for the corresponding layer. In the above, `Total Loop` represent the Convolution operation of and input of NCHW (1,3,112,112) with a kernel of output channels 64 and a size of 7x7.

The `Outer Loop` stores information on how many times the `Total Loop` needs to be iterated at the inner loop level. In the example above, the P of the `Total Loop` is 112, and the P of the `Inner Loop` is 23. Therefore, the `Outer Loop` should be set to 5 (e.g., ceiling(112/23)).

The `Inner Loop` represents the tile size that is loaded to the scratchpad memory. In the example above, input tiles of NCHW (1,3,23,22) and output tiles with 16 channels and a 7x7 kernel are configured to be used.
This example assumed that each element of the tensor is 4 Bytes.
Therefore, the size of the input tensor tile is 6,072 Bytes (=3x23x22), and the size of the weight tile is 12,544 Bytes (=4x16x7x7). The input and weight tiles are stored in the scratchpad memory, using 18616 Bytes. The size of the output tile is 17,408 Bytes (=16x17x16), and it is allocated in the accumulator.

**Note**: The size of the scratchpad memory and accumulator required by `Inner Loop` should not exceed half of the configured size for double-buffering.

This mapping is an example of the ResNet-18.

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

TBA
