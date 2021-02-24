set(DF_VERSION_STRING ${CURRENT_VERSION_STRING})
set(DF_VERSION_DATE ${CURRENT_VERSION_DATE})

set(DF_COPYRIGHT_STATEMENT "\
 Copyright (c) 2019-2021  G.E.T. department of SHINE project.\\n\
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
