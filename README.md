# NPU Simulator
## Requirements
* OS Distribution: Centos8 (Recommended)
### Python Package
* torch >= 1.10.1
* conan == 1.57.0
* onnxruntime >= 1.10.0

### Package
* cmake >= 3.22.1 (You need to build manually)
* gcc == 8.3


------------
## Mapping
ONNXim uses a hierarchical tiling method that can handle large tensor. To do this, it needs a proper mapping algorithm which can exploit memory efficiently. We've used the alogrithm that "Gemmini" uses.
Mapping hierarchy is composed to 3 parts.

1. Total Loop
2. Outer Loop
3. Inner Loop

### Total Loop 
`[T] N1 C3 M64 P112 Q112 S7 R7`
### Outer Loop
`[O] N1 C1 M4 P5 Q6 S1 R1`
### Inner Loop
`[I] N1 C3 M16 P23 Q22 S7 R7`

N: Batch size, C: Input channel, M: Output Channel, P: Output Rows, Q: Output Cols, S: Kernel Row, R: Kernel Cols

This mapping is an example of first convolution layer in ResNet18. Inner Loop is the tensor size that can be holded in scratch pad and accumulator.

Activation and Weight are stored in scratch pad, and output is stored in accumulator. This simulator consider the size of scratch pad as 256KB and accumulator size as 16KB (default, you can modify). Furthermore, it uses double buffering so that it could fill the scratch pad with half size. Mapping is calculated before implement simulator.


We provide mapping of ResNet18 in `models` directory, but it's not optimal. You can make the mapping on your way.

------------

## ONNX
You need to export your model to ONNX. We provide fused ResNet18 in `models` directory.

------------

## Hardware Configuration
`configs/*.json` In this directory, there are several json file that indicate HW config. You can modify the number of cores, sram size, dram type, and etc.

------------

# Getting Started
## method 1 (Docker Image)
```
$ git clone ssh://git@acpws-gitlab.postech.ac.kr:10023/hhk971/ai-framwork-sim.git
$ cd ai-framwork-sim
$ docker build . -t onnxim
```
build docker image

```
$ docker run -it onnxim
(docker) cd ai-framwork-sim
(docker) ./build/bin/Simulator --config ./configs/systolic_ws_8x8_c1_simple_noc.json --model ./models_list.json
```
run docker image


## method 2 (Mannual)
### Instrallation
```
$ git clone ssh://git@acpws-gitlab.postech.ac.kr:10023/hhk971/ai-framwork-sim.git
$ cd ai-framwork-sim
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

## Future Works
This version only supports GEMM and Conv Operation. We're developing the simulator to support other operations such as pooling.