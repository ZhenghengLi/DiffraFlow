df_standard_exe(ingester
    INCLUDE_DIRS /usr/include/hdf5/serial ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    LIBRARIES diffraflow::common diffraflow::algorithms
)
