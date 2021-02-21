project(combiner)

include_directories(
    inc
    ${dfcommon_INCLUDE_DIRS}
    )
add_definitions(${dfcommon_DEFINITIONS})

file(GLOB CC_FILES src/*.cc)

add_executable(${PROJECT_NAME} ${CC_FILES})
target_link_libraries(${PROJECT_NAME}
    dfcommon
    )

install(TARGETS ${PROJECT_NAME})

