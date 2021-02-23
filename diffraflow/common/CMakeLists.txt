df_standard_lib_project(dfcommon
    ALIAS diffraflow::common
    INCLUDE_DIRS /usr/include/hdf5/serial
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
)
