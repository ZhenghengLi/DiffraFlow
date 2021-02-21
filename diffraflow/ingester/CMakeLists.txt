df_standard_cpp_project(ingester
    INCLUDE_DIRS ${dfcommon_INCLUDE_DIRS} /usr/include/hdf5/serial
    DEFINITIONS ${dfcommon_DEFINITIONS}
    LIBRARIES dfcommon
)
