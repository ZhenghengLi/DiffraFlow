df_standard_exe(cuda-mean-rms-test
    INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    LIBRARIES diffraflow::common diffraflow::algorithms diffraflow::utilities CUDA::cudart
)
