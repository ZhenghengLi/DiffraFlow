# Dockerfile for Diffraflow project
# maintainer: Zhengheng Li <zhenghenge@gmail.com>

## build ####
FROM ubuntu:18.04 AS builder
# FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 AS builder
# install dependencies
RUN sed -i 's/archive.ubuntu.com/mirrors.huaweicloud.com/g' /etc/apt/sources.list && \
apt-get update && \
apt-get install -y --no-install-recommends \
openjdk-8-jdk build-essential \
libboost-dev \
libsnappy-dev liblog4cxx-dev \
libmsgpack-dev libzookeeper-mt-dev && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*
# build and install
ARG SOURCE_DIR=/opt/diffraflow_src
ARG INSTALL_DIR=/opt/diffraflow
ADD $PWD $SOURCE_DIR
RUN cd $SOURCE_DIR && \
./gradlew --no-daemon packageRelease && \
rm -rf $HOME/.gradle && \
mkdir $INSTALL_DIR && \
mv build/package_release/* $INSTALL_DIR && \
cd / && \
rm -rf $SOURCE_DIR

## deploy ####
FROM ubuntu:18.04
# FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
# install dependencies
RUN sed -i 's/archive.ubuntu.com/mirrors.huaweicloud.com/g' /etc/apt/sources.list && \
apt-get update && \
apt-get install -y --no-install-recommends \
openjdk-8-jre \
libsnappy-dev liblog4cxx-dev \
libmsgpack-dev libzookeeper-mt-dev \
netcat-openbsd && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*
# copy from builder
ARG INSTALL_DIR=/opt/diffraflow
COPY --from=builder $INSTALL_DIR $INSTALL_DIR
# add labels
ARG SOURCE_COMMIT
ARG COMMIT_MSG
LABEL description="High volume data acquisition and online data analysis for area detectors." \
maintainer="Zhengheng Li <zhenghenge@gmail.com>" \
source_commit="$SOURCE_COMMIT" \
commit_msg="$COMMIT_MSG"
# set runtime environment variables
ENV CLASSPATH=$INSTALL_DIR/jar/* \
PATH=$INSTALL_DIR/bin:$INSTALL_DIR/scripts:$PATH \
LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH \
SOURCE_COMMIT="$SOURCE_COMMIT" \
COMMIT_MSG="$COMMIT_MSG"
# use a non-root user
RUN useradd -m -u 42700 diffraflow
USER diffraflow
CMD ["/bin/bash"]

