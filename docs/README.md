# Documents of this project

## Installation

1. Install dependencies (Ubuntu 18.04+)  

   ```bash
   # Boost C++ Library, Google Snappy, LZ4, ZSTD, Apache log4cxx, MessagePack, ZooKeeper, C++ Rest SDK, HDF5
   sudo apt install libboost-all-dev libsnappy-dev liblz4-dev libzstd-dev liblog4cxx-dev libmsgpack-dev libzookeeper-mt-dev libcpprest-dev libhdf5-dev
   # Pulsar Client
   wget https://archive.apache.org/dist/pulsar/pulsar-2.5.0/DEB/apache-pulsar-client.deb
   sudo dpkg -i apache-pulsar-client.deb
   wget https://archive.apache.org/dist/pulsar/pulsar-2.5.0/DEB/apache-pulsar-client-dev.deb
   sudo dpkg -i apache-pulsar-client-dev.deb
   ```

2. Compile and install  

   ```bash
   cd DiffraFlow
   mkdir build
   cd build
   cmake ..
   cmake --build . --parallel $(nproc)
   cmake --install . --prefix <install-path>
   ```

## Architecture

The following picture shows the relationships between different components in the respect of data flow.

![architecture](images/architecture.png)
