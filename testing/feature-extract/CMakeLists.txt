df_standard_exe(feature-extract
    INCLUDE_DIRS /usr/include/hdf5/serial
    LIBRARIES diffraflow::common diffraflow::algorithms diffraflow::utilities CUDA::cudart
)
