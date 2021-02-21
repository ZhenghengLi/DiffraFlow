df_standard_cpp_project(generator
    INCLUDE_DIRS ${dfcommon_INCLUDE_DIRS} /usr/include/hdf5/serial
    DEFINITIONS ${dfcommon_DEFINITIONS}
    LIBRARIES dfcommon
)
