# Use Ubuntu 20.04 as the base image, where GCC 10 is available
FROM ubuntu:20.04

# Avoid prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Update and install software
RUN apt-get update && apt-get install -y \
    gcc-10 g++-10 python3.8 python3-pip git wget make \
    libssl-dev libasan5 libubsan1

# Set GCC 10 as the default gcc and g++ compilers
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100

# Set the working directory
WORKDIR /workspace

# Install CMake 3.22.0 from source
RUN wget https://github.com/Kitware/CMake/releases/download/v3.22.0/cmake-3.22.0.tar.gz && \
    tar -xvzf cmake-3.22.0.tar.gz && \
    cd cmake-3.22.0 && \
    ./bootstrap && \
    make -j$(nproc) && \
    make install

# Install specific Python packages with pip
RUN pip3 install conan==1.57.0 transformers==4.40.1 onnx onnxruntime torch==2.3.1 torchvision optimum

# Copy your project files into the image
COPY ./ ONNXim

# Prepare ONNXim project
RUN cd ONNXim && \
    git submodule update --recursive --init && \
    mkdir -p build && \
    cd build && \
    conan install .. --build=missing && \
    cmake .. && \
    make -j$(nproc)

# Set environment variable
ENV ONNXIM_HOME /workspace/ONNXim

# Final command
CMD ["echo", "Welcome to ONNXim!"]

