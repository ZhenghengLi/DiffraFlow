FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

LABEL description="High volume data acquisition and online data analysis for area detectors."
LABEL maintainer="Zhengheng Li <zhenghenge@gmail.com>"

# set buildtime variables
ARG source_dir=/opt/diffraflow_src
ARG install_dir=/opt/diffraflow

# install dependencies
# RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
RUN apt-get update && \
apt-get install -y --no-install-recommends \
openjdk-8-jdk build-essential \
libboost-system-dev libboost-log-dev \
libsnappy-dev && \
apt-get clean

# build and install
ADD $PWD $source_dir
RUN cd $source_dir && \
./gradlew --no-daemon packageRelease && \
rm -r $HOME/.gradle && \
mkdir $install_dir && \
cp -r build/package_release/* $install_dir && \
cd / && \
rm -r $source_dir

# set runtime environment variables
ENV CLASSPATH=$install_dir/jar/*
ENV PATH=$install_dir/bin:$PATH
ENV LD_LIBRARY_PATH=$install_dir/lib:$LD_LIBRARY_PATH
VOLUME ["/workspace"]

CMD ["/bin/bash"]

