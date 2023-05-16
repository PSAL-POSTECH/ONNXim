FROM centos:8
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-Linux-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-Linux-*

RUN yum update -y && yum install -y python3 python3-pip git wget gcc gcc-c++ make openssl-devel  

WORKDIR /workspace

RUN wget https://github.com/Kitware/CMake/releases/download/v3.22.0/cmake-3.22.0.tar.gz
RUN tar -xvzf cmake-3.22.0.tar.gz && cd cmake-3.22.0 && ./bootstrap && make -j && make install
RUN pip3 install conan==1.57.0
COPY ./ ai-framwork-sim
RUN cd ai-framwork-sim && git submodule update --recursive --init && mkdir build && cd build && conan install .. && cmake .. && make -j

RUN echo "Welcome to ONNXim!"
