#ifndef __schedtools_h__
#define __schedtools_h__

#include <string>
#include <sched.h>

using std::string;

namespace diffraflow {
    namespace schedtools {

        int string_to_cpu_set(cpu_set_t* cpu_set, const string& cpu_list);

    } // namespace schedtools
} // namespace diffraflow

#endif
