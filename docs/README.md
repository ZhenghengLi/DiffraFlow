# Documents of this project

## Installation

1. Install dependencies  
   * For Ubuntu:  

   ```bash
    # Boost C++ Library, Google Snappy, Apache log4cxx, MessagePack, ZooKeeper
    sudo apt install libboost-dev libsnappy-dev liblog4cxx-dev libmsgpack-dev libzookeeper-mt-dev
   ```

   * For Mac OS:  

   ```bash
    # Boost C++ Library, Google Snappy, Apache log4cxx, MessagePack, ZooKeeper
    brew install boost snappy log4cxx msgpack zookeeper
   ```

2. Compile and install  
   * For debug variant:

   ```bash
    ./gradlew build
   ```

   then all binaries of debug variant will be compiled and installed in build/package_debug.  

   * For release variant:

   ```bash
    ./gradlew packageRelease
   ```

   then all binaries of release variant will be compiled and installed in build/package_release.

## Architecture

This picture shows the architecture designing details of the data flow from source to sink and the strategy for online analysis.

![architecture](images/architecture.png)

