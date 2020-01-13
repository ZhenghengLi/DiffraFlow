# Documents of this project

## Installation

1. Install dependencies  
   * For Ubuntu:  

   ```bash
    # Boost C++ Library
    sudo apt install libboost-all-dev
    # Google Snappy
    sudo apt install libsnappy-dev
   ```

   * For Mac OS:  

   ```bash
    # Boost C++ Library
    brew install boost
    # Google Snappy
    brew install snappy
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
