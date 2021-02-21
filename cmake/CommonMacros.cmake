function(df_standard_cpp_project proj_name)
    project(${proj_name})
    set(multiValueArgs INCLUDE_DIRS DEFINITIONS LIBRARIES)
    cmake_parse_arguments(ARG "" "" "${multiValueArgs}" ${ARGN})
    include_directories(
        inc
        ${ARG_INCLUDE_DIRS}
    )
    add_definitions(
        ${ARG_DEFINITIONS}
    )
    file(GLOB CC_FILES src/*.cc)
    add_executable(${PROJECT_NAME} ${CC_FILES})
    target_link_libraries(${PROJECT_NAME}
        ${ARG_LIBRARIES}
    )
    install(TARGETS ${PROJECT_NAME})
endfunction()

function(df_standard_lib_project proj_name)
    project(${proj_name})
    set(multiValueArgs INCLUDE_DIRS DEFINITIONS LIBRARIES)
    cmake_parse_arguments(ARG "" "" "${multiValueArgs}" ${ARGN})
    include_directories(
        inc
        ${ARG_INCLUDE_DIRS}
    )
    add_definitions(
        ${ARG_DEFINITIONS}
    )
    file(GLOB CC_FILES src/*.cc)
    add_library(${PROJECT_NAME} SHARED ${CC_FILES})
    target_link_libraries(${PROJECT_NAME}
        ${ARG_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
    )
    install(TARGETS ${PROJECT_NAME})
    set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/inc CACHE INTERNAL "" FORCE)
    set(${PROJECT_NAME}_DEFINITIONS ${ARG_DEFINITIONS} CACHE INTERNAL "" FORCE)
endfunction()