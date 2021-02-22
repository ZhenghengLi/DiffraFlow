function(df_standard_cpp_project proj_name)
    set(options CUDA)
    set(multiValueArgs INCLUDE_DIRS DEFINITIONS LIBRARIES)
    cmake_parse_arguments(ARG "${options}" "" "${multiValueArgs}" ${ARGN})

    if(ARG_CUDA)
        project(${proj_name} C CXX CUDA)
    else()
        project(${proj_name} C CXX)
    endif()

    include_directories(
        inc
        ${ARG_INCLUDE_DIRS}
    )
    add_definitions(
        ${ARG_DEFINITIONS}
    )

    file(GLOB sources src/*)
    file(GLOB headers inc/*)
    add_executable(${PROJECT_NAME} ${sources} ${headers})
    target_link_libraries(${PROJECT_NAME}
        ${ARG_LIBRARIES}
    )

    install(TARGETS ${PROJECT_NAME} DESTINATION bin)
endfunction()

function(df_standard_lib_project proj_name)
    set(options CUDA)
    set(multiValueArgs INCLUDE_DIRS DEFINITIONS LIBRARIES)
    cmake_parse_arguments(ARG "${options}" "" "${multiValueArgs}" ${ARGN})

    if(ARG_CUDA)
        project(${proj_name} C CXX CUDA)
    else()
        project(${proj_name} C CXX)
    endif()

    include_directories(
        inc
        ${ARG_INCLUDE_DIRS}
    )
    add_definitions(
        ${ARG_DEFINITIONS}
    )

    file(GLOB sources src/*)
    file(GLOB headers inc/*)
    add_library(${PROJECT_NAME} SHARED ${sources} ${headers})
    target_link_libraries(${PROJECT_NAME}
        ${ARG_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
    )

    install(TARGETS ${PROJECT_NAME} DESTINATION lib)

    set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/inc CACHE INTERNAL "" FORCE)
    set(${PROJECT_NAME}_DEFINITIONS ${ARG_DEFINITIONS} CACHE INTERNAL "" FORCE)
endfunction()

