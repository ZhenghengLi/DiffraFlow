function(df_standard_cpp_project proj_name)
    set(multiValueArgs INCLUDE_DIRS DEFINITIONS LIBRARIES)
    cmake_parse_arguments(ARG "" "" "${multiValueArgs}" ${ARGN})

    project(${proj_name})

    file(GLOB sources src/*)
    file(GLOB headers inc/*)
    add_executable(${PROJECT_NAME} ${sources} ${headers})

    target_link_libraries(${PROJECT_NAME}
        ${ARG_LIBRARIES}
    )
    target_include_directories(${PROJECT_NAME} PRIVATE inc ${ARG_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME} PRIVATE ${ARG_DEFINITIONS})

    install(TARGETS ${PROJECT_NAME} DESTINATION bin)
endfunction()

function(df_standard_lib_project proj_name)
    set(multiValueArgs INCLUDE_DIRS DEFINITIONS LIBRARIES)
    cmake_parse_arguments(ARG "" "" "${multiValueArgs}" ${ARGN})

    project(${proj_name})

    file(GLOB sources src/*)
    file(GLOB headers inc/*)
    file(GLOB publics pub/*)
    add_library(${PROJECT_NAME} SHARED ${sources} ${headers} ${publics})

    target_link_libraries(${PROJECT_NAME}
        ${ARG_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
    )
    target_include_directories(${PROJECT_NAME} PUBLIC  pub)
    target_include_directories(${PROJECT_NAME} PRIVATE inc ${ARG_INCLUDE_DIRS})
    target_compile_definitions(${PROJECT_NAME} PRIVATE ${ARG_DEFINITIONS})

    install(TARGETS ${PROJECT_NAME} DESTINATION lib)
endfunction()

