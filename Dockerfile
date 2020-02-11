# Dockerfile for Diffraflow project
# maintainer: Zhengheng Li <zhenghenge@gmail.com>

## build ####
FROM ubuntu:18.04 AS builder
# FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 AS builder
ARG source_dir=/opt/diffraflow_src
ARG install_dir=/opt/diffraflow
# install dependencies
# RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
RUN apt-get update && \
apt-get install -y --no-install-recommends \
openjdk-8-jdk build-essential \
libboost-dev \
libsnappy-dev liblog4cxx-dev && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*
# build and install
ADD $PWD $source_dir
RUN cd $source_dir && \
./gradlew --no-daemon packageRelease && \
rm -rf $HOME/.gradle && \
mkdir $install_dir && \
cp -r build/package_release/* $install_dir && \
cd / && \
rm -rf $source_dir

## deploy ####
FROM ubuntu:18.04
# FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
ARG install_dir=/opt/diffraflow
LABEL description="High volume data acquisition and online data analysis for area detectors." \
maintainer="Zhengheng Li <zhenghenge@gmail.com>"
# install dependencies
# RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
RUN apt-get update && \
apt-get install -y --no-install-recommends \
openjdk-8-jre \
libsnappy-dev liblog4cxx-dev && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*
# copy from builder
COPY --from=builder $install_dir $install_dir
# set runtime environment variables
ENV CLASSPATH=$install_dir/jar/* \
PATH=$install_dir/bin:$PATH \
LD_LIBRARY_PATH=$install_dir/lib:$LD_LIBRARY_PATH
VOLUME ["/workspace"]
CMD ["/bin/bash"]

