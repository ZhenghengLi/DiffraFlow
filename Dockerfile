# Dockerfile for Diffraflow project
# maintainer: Zhengheng Li <zhenghenge@gmail.com>

# build ############################################################
FROM ubuntu:18.04 AS builder
# FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 AS builder

# install dependencies
ARG PULSAR_VERSION=2.5.0
ARG PULSAR_URL_PREFIX=https://archive.apache.org/dist/pulsar/pulsar-${PULSAR_VERSION}
ARG PULSAR_CLIENT=apache-pulsar-client.deb
ARG PULSAR_CLIENT_DEV=apache-pulsar-client-dev.deb
RUN set -x \
&& sed -i 's/archive.ubuntu.com/mirrors.huaweicloud.com/g' /etc/apt/sources.list \
## install by apt-get
&& apt-get update \
&& apt-get install -y --no-install-recommends \
    ca-certificates \
    openjdk-8-jdk build-essential \
    libboost-dev libboost-filesystem-dev \
    liblz4-dev libsnappy-dev libzstd-dev liblog4cxx-dev \
    libmsgpack-dev libzookeeper-mt-dev libcpprest-dev libhdf5-dev \
    wget \
## install pulsar c++ client
&& mkdir -pv /tmp/pulsar/DEB \
&& cd /tmp/pulsar \
&& wget -q "${PULSAR_URL_PREFIX}/DEB/${PULSAR_CLIENT}" -P "DEB" \
&& wget -q "${PULSAR_URL_PREFIX}/DEB/${PULSAR_CLIENT}.sha512" -P "DEB" \
&& sha512sum -c "DEB/${PULSAR_CLIENT}.sha512" \
&& dpkg -i "DEB/${PULSAR_CLIENT}" \
&& wget -q "${PULSAR_URL_PREFIX}/DEB/${PULSAR_CLIENT_DEV}" -P "DEB" \
&& wget -q "${PULSAR_URL_PREFIX}/DEB/${PULSAR_CLIENT_DEV}.sha512" -P "DEB" \
&& sha512sum -c "DEB/${PULSAR_CLIENT_DEV}.sha512" \
&& dpkg -i "DEB/${PULSAR_CLIENT_DEV}" \
&& cd / \
&& rm -rf /tmp/pulsar \
## clean
&& apt-get autoremove -y wget \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

# build and install
ARG SOURCE_DIR=/opt/diffraflow_src
ARG INSTALL_DIR=/opt/diffraflow
ADD $PWD $SOURCE_DIR
RUN set -x \
## build
&& cd $SOURCE_DIR \
&& ./gradlew --no-daemon packageRelease \
## install
&& mkdir $INSTALL_DIR \
&& mv build/package_release/* $INSTALL_DIR \
## clean
&& cd / \
&& rm -rf $HOME/.gradle \
&& rm -rf $SOURCE_DIR

# deploy ############################################################
FROM ubuntu:18.04
# FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

# install dependencies
ARG PULSAR_VERSION=2.5.0
ARG PULSAR_URL_PREFIX=https://archive.apache.org/dist/pulsar/pulsar-${PULSAR_VERSION}
ARG PULSAR_CLIENT=apache-pulsar-client.deb
RUN set -x \
&& sed -i 's/archive.ubuntu.com/mirrors.huaweicloud.com/g' /etc/apt/sources.list \
## install by apt-get
&& apt-get update \
&& apt-get install -y --no-install-recommends \
    ca-certificates \
    openjdk-8-jre \
    libboost-dev libboost-filesystem-dev \
    liblz4-dev libsnappy-dev libzstd-dev liblog4cxx-dev \
    libmsgpack-dev libzookeeper-mt-dev libcpprest-dev libhdf5-dev \
    netcat-openbsd \
    wget \
## install pulsar c++ client
&& mkdir -pv /tmp/pulsar/DEB \
&& cd /tmp/pulsar \
&& wget -q "${PULSAR_URL_PREFIX}/DEB/${PULSAR_CLIENT}" -P "DEB" \
&& wget -q "${PULSAR_URL_PREFIX}/DEB/${PULSAR_CLIENT}.sha512" -P "DEB" \
&& sha512sum -c "DEB/${PULSAR_CLIENT}.sha512" \
&& dpkg -i "DEB/${PULSAR_CLIENT}" \
&& cd / \
&& rm -rf /tmp/pulsar \
## clean
&& apt-get autoremove -y wget \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

# copy from builder
ARG INSTALL_DIR=/opt/diffraflow
COPY --from=builder $INSTALL_DIR $INSTALL_DIR

# add labels
ARG SOURCE_COMMIT
ARG COMMIT_MSG
ARG BUILD_TIME
LABEL description="High volume data acquisition and online data analysis for area detectors." \
maintainer="Zhengheng Li <zhenghenge@gmail.com>" \
source_commit="$SOURCE_COMMIT" \
commit_msg="$COMMIT_MSG" \
build_time="$BUILD_TIME"

# set runtime environment variables
ENV CLASSPATH=$INSTALL_DIR/jar/* \
PATH=$INSTALL_DIR/bin:$INSTALL_DIR/scripts:$PATH \
LD_LIBRARY_PATH=$INSTALL_DIR/lib:$LD_LIBRARY_PATH \
SOURCE_COMMIT="$SOURCE_COMMIT" \
COMMIT_MSG="$COMMIT_MSG" \
BUILD_TIME="$BUILD_TIME"

# user setting
RUN set -x \
## set root password for runtime debug
&& echo "root:20180427" | chpasswd \
## use a non-root user for normal work
&& groupadd diffraflow --gid=42700 \
&& useradd -m -g diffraflow --uid=42700 diffraflow -s /bin/bash

USER diffraflow

CMD ["/bin/bash"]
