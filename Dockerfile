# Dockerfile for Diffraflow project
# maintainer: Zhengheng Li <zhenghenge@gmail.com>

# build ############################################################
FROM zhenghengli/ubuntu-devel:20.04 AS builder

# build and install
ARG SOURCE_DIR=/opt/diffraflow_src
ARG BUILD_DIR=/opt/diffraflow_build
ARG INSTALL_DIR=/opt/diffraflow
ADD $PWD $SOURCE_DIR
RUN set -x \
    ## build and install
    && cmake -S $SOURCE_DIR -B $BUILD_DIR \
    && cmake --build $BUILD_DIR \
    && cmake --install $BUILD_DIR --prefix $INSTALL_DIR \
    ## clean
    && rm -rf $SOURCE_DIR \
    && rm -rf $BUILD_DIR

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
ENV PATH=$INSTALL_DIR/bin:$INSTALL_DIR/scripts:$PATH \
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
