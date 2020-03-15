FROM openjdk:8-jre-slim
ARG ZK_VERSION=3.6.0
ARG DISTRO_NAME=apache-zookeeper-${ZK_VERSION}-bin
ARG DISTRO_URL=https://archive.apache.org/dist/zookeeper/zookeeper-${ZK_VERSION}/${DISTRO_NAME}.tar.gz
RUN set -x \
# install dependencies
&& apt-get update \
&& apt-get install -y --no-install-recommends ca-certificates gosu gnupg netcat wget \
# verify that gosu binary works
&& gosu nobody true \
# download
&& mkdir -pv /opt/tmp\
&& cd /opt/tmp \
&& wget -q "${DISTRO_URL}" \
&& wget -q "${DISTRO_URL}.asc" \
&& wget -q "${DISTRO_URL}.sha512" \
&& sha512sum -c ${DISTRO_NAME}.tar.gz.sha512 \
&& wget https://dist.apache.org/repos/dist/release/zookeeper/KEYS \
&& gpg --import KEYS \
&& gpg --batch --verify "$DISTRO_NAME.tar.gz.asc" "$DISTRO_NAME.tar.gz" \
&& tar -xzf "$DISTRO_NAME.tar.gz" \
&& mv ${DISTRO_NAME} .. \
&& cd .. \
&& rm -rf tmp \
# clean
&& apt-get autoremove -y wget gnupg \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/* \
CMD ["bash"]