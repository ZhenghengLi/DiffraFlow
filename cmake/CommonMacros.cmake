function(df_standard_exe target_name)
    set(multiValueArgs INCLUDE_DIRS DEFINITIONS LIBRARIES)
    cmake_parse_arguments(ARG "" "" "${multiValueArgs}" ${ARGN})

    file(GLOB sources src/*)
    file(GLOB headers inc/*)
    add_executable(${target_name} ${sources} ${headers})

    target_include_directories(${target_name} PRIVATE inc ${ARG_INCLUDE_DIRS})
    target_compile_definitions(${target_name} PRIVATE ${ARG_DEFINITIONS})
    target_link_libraries(${target_name} PRIVATE ${ARG_LIBRARIES})

    install(TARGETS ${target_name} DESTINATION bin)
endfunction()

function(df_standard_lib target_name)
    set(oneValueArgs ALIAS)
    set(multiValueArgs INCLUDE_DIRS DEFINITIONS LIBRARIES)
    cmake_parse_arguments(ARG "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    file(GLOB sources src/*)
    file(GLOB headers inc/*)
    file(GLOB publics pub/*)
    add_library(${target_name} SHARED ${sources} ${headers} ${publics})

    target_include_directories(${target_name} PUBLIC  pub)
    target_include_directories(${target_name} PRIVATE inc ${ARG_INCLUDE_DIRS})
    target_compile_definitions(${target_name} PRIVATE ${ARG_DEFINITIONS})
    target_link_libraries(${target_name} PUBLIC ${ARG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

    if(DEFINED ARG_ALIAS)
        add_library(${ARG_ALIAS} ALIAS ${target_name})
    endif()

    install(TARGETS ${target_name} DESTINATION lib)
endfunction()

