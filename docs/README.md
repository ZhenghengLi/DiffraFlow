# Documents of this project

## Installation

1. Install dependencies  
   * For Ubuntu:  

   ```bash
    # Boost C++ Library, Google Snappy, Apache log4cxx, MessagePack
    sudo apt install libboost-dev libsnappy-dev liblog4cxx-dev libmsgpack-dev
   ```

   * For Mac OS:  

   ```bash
    # Boost C++ Library, Google Snappy, Apache log4cxx, MessagePack
    brew install boost snappy log4cxx msgpack
   ```

2. Compile and install  
   * For debug version:

   ```bash
    ./gradlew build
   ```

   then all binaries will be installed in build/package_debug.  

   * For release version:

   ```bash
    ./gradlew packageRelease
   ```

   then all binaries will be installed in build/package_release.

## Architecture

This picture shows the architecture designing details of the data flow from source to sink and the strategy for online analysis.

![architecture](images/architecture.png)

