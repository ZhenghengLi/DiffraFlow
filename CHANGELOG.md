# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

(empty)

## [0.0.2] - 2020-01-08

### Changed

- 'dispatcher' is rewritten by C++, and the Java version is removed.
- Use an unified data exchange protocol between different components, and the protocol is defined by docs/protocol.txt.
- The data transferred from 'dispatcher' to 'combiner' can be compressed, but now the compression code is not yet added.

## [0.0.1] - 2019-10-31

### Added

- Add this changelog file.
