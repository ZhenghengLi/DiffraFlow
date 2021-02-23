cmake_minimum_required(VERSION 3.10)

project(DiffraFlow
    DESCRIPTION "High throughput streaming data acquisition and online analysis for pixel array detectors."
    LANGUAGES C CXX CUDA
    )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_FLAGS "-gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/package)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(VersionInfo)
include(CommonMacros)

find_package(Threads REQUIRED)

add_subdirectory(diffraflow)
add_subdirectory(testing)

# install script files
set(SCRIPT_FILES
    scripts/thisenv.sh
    scripts/check_cnxn_address.sh
    scripts/check_cnxn_addrlist.sh
    scripts/cxidb_euxfel_utils.py
    scripts/calib_gen.py
    scripts/alignment_index_proc.py
    scripts/event_select_proc.py
    scripts/raw_data_gen.py
    scripts/copy_sender_data.sh
    )
install(PROGRAMS
    ${SCRIPT_FILES}
    DESTINATION scripts
    )

# install config files
set(CONFIG_FILES
    config/combiner_1.conf
    config/dispatcher_1.conf
    config/combiner_addr_list.txt
    config/ingester.conf
    config/ingester.dyconf
    config/log4cxx.properties
    config/zookeeper_k8s.conf
    )
install(FILES
    ${CONFIG_FILES}
    DESTINATION conf
    )
