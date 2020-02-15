# Documents of this project

## Installation

1. Install dependencies  
   * For Ubuntu:  

   ```bash
    # Boost C++ Library
    sudo apt install libboost-dev
    # Google Snappy
    sudo apt install libsnappy-dev
    # log4cxx
    sudo apt install liblog4cxx-dev
   ```

   * For Mac OS:  

   ```bash
    # Boost C++ Library
    brew install boost
    # Google Snappy
    brew install snappy
    # log4cxx
    brew install log4cxx
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

