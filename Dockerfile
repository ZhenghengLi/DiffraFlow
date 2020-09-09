# Dockerfile for Diffraflow project
# maintainer: Zhengheng Li <zhenghenge@gmail.com>

# build ############################################################
FROM zhenghengli/ubuntu-devel:20.04 AS builder

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
FROM zhenghengli/ubuntu-runtime:20.04

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
    && groupadd diffraflow --gid=1010 \
    && useradd -m -g diffraflow --uid=1017 diffraflow -s /bin/bash

USER diffraflow

CMD ["/bin/bash"]
