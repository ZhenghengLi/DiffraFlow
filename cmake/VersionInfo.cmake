set(DF_VERSION_STRING "v0.2.0")
set(DF_VERSION_DATE "2021-02-22")

set(DF_COPYRIGHT_STATEMENT "\
 Copyright (c) 2019-2020  G.E.T. department of SHINE project.\\n\
 All Rights Reserved."
)

set(DF_PROJECT_URL "<url>")
set(DF_MAIN_CONTRIBUTORS "\
 - Zhengheng Li <lizhh1@shanghaitech.edu.cn>"
)

find_package(Git)
if(Git_FOUND AND EXISTS ${CMAKE_SOURCE_DIR}/.git)
    execute_process(COMMAND ${GIT_EXECUTABLE} --git-dir=${CMAKE_SOURCE_DIR}/.git describe --always
        OUTPUT_VARIABLE git_describe_always
        RESULT_VARIABLE git_describe_errorcode
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    execute_process(COMMAND ${GIT_EXECUTABLE} --git-dir=${CMAKE_SOURCE_DIR}/.git show -s --format=%cd --date=short
        OUTPUT_VARIABLE git_show_date
        RESULT_VARIABLE git_show_errorcode
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    if(NOT git_describe_errorcode AND NOT git_show_errorcode)
        set(DF_VERSION_STRING ${git_describe_always})
        set(DF_VERSION_DATE ${git_show_date})
    endif()
endif()

message("version: ${DF_VERSION_STRING}")
message("date: ${DF_VERSION_DATE}")