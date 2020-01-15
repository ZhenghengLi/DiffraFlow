FROM ubuntu:18.04

# install dependencies
RUN apt-get update
RUN apt-get install -y openjdk-8-jdk build-essential libboost-all-dev libsnappy-dev

ADD $PWD /diffraflow_src
WORKDIR /diffraflow_src
RUN ./gradlew packageRelease
RUN mkdir /opt/diffraflow
RUN cp -r build/package_release/* /opt/diffraflow

WORKDIR /
RUN rm -r /diffraflow_src

CMD ["/bin/bash"]

