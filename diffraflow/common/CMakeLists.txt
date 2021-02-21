project(dfcommon)

include_directories(
    inc
    /usr/include/hdf5/serial
    )
add_definitions(-DBOOST_ALL_DYN_LINK)

file(GLOB CC_FILES src/*.cc)

add_library(${PROJECT_NAME} SHARED ${CC_FILES})
target_link_libraries(${PROJECT_NAME}
    pulsar
    log4cxx
    zookeeper_mt
    cpprest ssl
    crypto
    hdf5_cpp
    hdf5_serial
    boost_filesystem
    boost_system
    ${CMAKE_THREAD_LIBS_INIT}
    )

install(TARGETS ${PROJECT_NAME})

set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/inc CACHE INTERNAL "" FORCE)
set(${PROJECT_NAME}_DEFINITIONS -DBOOST_ALL_DYN_LINK CACHE INTERNAL "" FORCE)

