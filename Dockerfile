FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

LABEL description="High volume data acquisition and online data analysis for area detectors."

# set buildtime variables
ARG source_dir=/opt/diffraflow_src
ARG install_dir=/opt/diffraflow

# install dependencies
# RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y openjdk-8-jdk build-essential libboost-all-dev libsnappy-dev

# build and install
ADD $PWD $source_dir
WORKDIR $source_dir
RUN ./gradlew packageRelease
RUN mkdir $install_dir
RUN cp -r build/package_release/* $install_dir

# remove source code
WORKDIR /
RUN rm -r $source_dir
VOLUME ["/workspace"]

# set environment variables for runtime
ENV CLASSPATH=$install_dir/jar/*
ENV PATH=$install_dir/bin:$PATH
ENV LD_LIBRARY_PATH=$install_dir/lib:$LD_LIBRARY_PATH

CMD ["/bin/bash"]

