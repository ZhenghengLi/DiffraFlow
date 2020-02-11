# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### TODO

- [x] switch to use log4cxx from boost.log
- [ ] resolve the path of combiner_addr_list.txt
- [ ] OptionManager for parsing command line parameters

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
