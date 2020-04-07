# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### TODO

- [x] write the template code to do data processing pipeline in ingester.
- [ ] write the class to sink data into disk.

## [0.0.10] - 2020-04-05

## Changed

- DynamicConfiguration: use JSON instead of MessagePack to serialize configuration data stored in ZooKeeper.
- GenericServer: use async and future to start and stop server, this make it convenient to run multiple servers in one executable.
- BlockingQueue: data consuming functions return false only when the internal queue is empty, even if the queue is stopped.

## Added

- CmbImgCache: the code to do time alignment is added, but only tested with single-detector data.
- The data exchanging logic between combiner and ingester is added and tested, and it is ready to write data processing pipeline in ingester.

## [0.0.9] - 2020-03-26

### Changed

- Data transerring code using socket is optimized. The related classes include GenericClient, GenericConnection and DspSender.
- Use MSGPACK_DEFINE_MAP to do serialization using MessagePack, which will serialize C++ object into JSON object and keep the variable name in the serialized data.

### Added

- Add code to report runtime metrics. Two methods are applied. One is using Pulsar C++ client to periodically send runtime metrics to a Pulsar message queue cluster. The other one is using [C++ REST SDK](https://github.com/microsoft/cpprestsdk) to start a http server, which can reply the latest runtime metrics on the GET request from client. The data of runtime metrics is aggregated and serialized by JSON.

## [0.0.8] - 2020-03-18

### Added

- Add [LZ4](https://github.com/lz4/lz4) and [ZSTD](https://github.com/facebook/zstd) compression support for the data transferring from **dispatcher** to **combiner**.
- Add deploy folder, which contains the Kuberenetes deployment YAML files for all components and services. The deployment of **dispatcher** and **combiner** on a local vm-built Kubernetes cluster was tested, and also for that of ZooKeeper, BookKeeper and Pulsar.
- Install Apache Pulsar C++ client in Dockerfile, as Pulsar is planned to be used for system runtime monitoring.

## [0.0.7] - 2020-02-29

### Changed

- use MessagePack library instead self-writting code to serialize ImageFrame and ImageData.

### Added

- Add class DynamicConfiguration which can dynamically update configurations at runtime. This class works based on ZooKeeper.
- Add **controller** which is responsible for updating the configurations of **ingester** and **monitor** in ZooKeeper at runtime, and fetching online analysis results from **monitor** in a round-robin way on the request from front-end client.

## [0.0.6] - 2020-02-12

### Changed

- log library is switched from boost.log to log4cxx.
- use a specific OptionsManager to parse command line parameters
- dispatcher: combiner addresses file is now set by a command line option instead of inside configuration file.

## [0.0.5] - 2020-01-16

### Changed

- The Dockerfile is optimized, and the CI workflow is setup.

## [0.0.4] - 2020-01-15

### Added

- GenericServer now can listen on a specific host.
- Add Dockerfile and start to use Docker for automated building.

## [0.0.3] - 2020-01-14

### Added

- The code (using [Snappy](https://github.com/google/snappy)) to compress the data transferred from **dispatcher** to **combiner** is added.
- Data compression between **dispatcher** and **combiner** can be switched on or off by **dispatcher**'s configuration file using *compress_flag*.

### Changed

- Serialization: use try-catch clause to check exceptions instead of using lots of if-else.

## [0.0.2] - 2020-01-08

### Changed

- 'dispatcher' is rewritten by C++, and the Java version is removed.
- Use an unified data exchange protocol between different components, and the protocol is defined by docs/protocol.txt.
- The data transferred from 'dispatcher' to 'combiner' can be compressed, but now the compression code is not yet added.

## [0.0.1] - 2019-10-31

### Added

- Add this changelog file.
