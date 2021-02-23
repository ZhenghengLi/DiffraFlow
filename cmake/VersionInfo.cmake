set(DF_COPYRIGHT_STATEMENT "\
 Copyright (c) 2019-2020  G.E.T. department of SHINE project.\\n\
 All Rights Reserved."
)

set(DF_PROJECT_URL "<url>")
set(DF_MAIN_CONTRIBUTORS "\
 - Zhengheng Li <lizhh1@shanghaitech.edu.cn>"
)

set(DF_VERSION_STRING "v0.2.0")
set(DF_VERSION_DATE "2021 Feb 23")

function(change_var variable value)
    set(${variable} ${value} PARENT_SCOPE)
endfunction()

