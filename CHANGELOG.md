# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### TODO

- [x] change the data structure of ImageFrame and adjust all the related code. (finished but not tested)
- [ ] implement sender and trigger.
- [ ] aggregator: running on the pulsar consumer side to aggregate metrics from all components, calculate speeds and serve results via HTTP GET.

## [0.0.13] - 2020-07-15

### Changed

- Dockerfile: switched to use Pulsar v2.5.2 instead of v2.5.0.
- **combiner** now will force doing alignment if the module data in cache stays for a relatively long time. This is useful to clear the cache after a run finished but there is still some partial data remained in cache.

### Added

- Added file .clang-format and applied clang-format for all C++ code.
- Added the YAML deployment files for **ingester**, **monitor** and **controller**, and together with **dispatcher** and **combiner**, the deployments of all these 5 components on demo machine are tested.
- Added some programs to convert the EuXFEL data published on CXIDB for the purpose of using it as the test data for this project. These programs include scripts/{calib_gen.py, alignment_index_proc.py, event_select_proc.py, event_view_proc.py, raw_data_gen.py} and testing/generator. And the raw data format is defined by file docs/rawdata-format.txt.
- Added some kubernetes job deployment files for doing the data generating and converting works on demo machine.

## [0.0.12] - 2020-04-30

### Added

- Finished the code of **monitor** to fetch image data from **ingester** by an HTTP client, as well as the online data analysis template code, the HTTP server to provide image data and analysis results on request.
- Finished the load balancer in **controller** for multiple monitors running on different nodes, and the support for ZooKeeper operations via RESTful API, including HTTP verbs GET, POST, PUT, PATCH and DELETE.
- Added the code to collect metrics for **combiner**, **ingester** and **monitor**.

## [0.0.11] - 2020-04-17

### Changed

- DynamicConfiguration: update, delete and fetch operations of config_map on ZooKeeper now support version check.

### Added

- Finished the template code to do data processing pipeline in **ingester**, and the specific data processing algorithms in each step of the pipeline need to be implemented later.
- Added classes to read/write image data from/into HDF5 files, in which the SWMR mode is supported. Those classes are used to sink data into storage in **ingester**.
- **ingester** starts an HTTP server when it starts to provide the latest image data on the request from **monitor**.

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
