df_standard_lib(dfcommon
    ALIAS diffraflow::common
    INCLUDE_DIRS /usr/include/hdf5/serial ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    LIBRARIES
        pulsar
        log4cxx
        zookeeper_mt
        cpprest
        ssl
        crypto
        hdf5_cpp
        hdf5_serial
        boost_filesystem
        boost_system
        CUDA::cudart
        Threads::Threads
)

configure_file(inc/VersionInfo.hh.in inc/VersionInfo.hh)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/inc)

