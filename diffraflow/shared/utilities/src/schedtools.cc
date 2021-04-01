#include "schedtools.hh"

#include <vector>
#include <set>
#include <boost/algorithm/string.hpp>
#include <sys/sysinfo.h>

using std::vector;
using std::set;
using std::stoi;

int diffraflow::schedtools::string_to_cpu_set(cpu_set_t* cpu_set, const string& cpu_list) {
    if (cpu_set == nullptr) return 1; // empty pointer

    vector<string> cpu_range_list;
    boost::split(cpu_range_list, cpu_list, boost::is_any_of(","));
    if (cpu_range_list.empty()) return 2; // empty string

    int total_cpu_count = get_nprocs();
    set<int> cpu_set_list;
    for (const string& cpu_range : cpu_range_list) {
        vector<string> cpus;
        boost::split(cpus, cpu_range, boost::is_any_of("-"));
        if (cpus.size() == 1) {
            int cpu_id = stoi(cpus[0]);
            if (cpu_id >= total_cpu_count) return 3; // large cpu id
            cpu_set_list.insert(cpu_id);
        } else if (cpus.size() == 2) {
            int cpu_begin = stoi(cpus[0]);
            int cpu_end = stoi(cpus[1]);
            if (cpu_begin > cpu_end) return 4;        // wrong range
            if (cpu_end >= total_cpu_count) return 5; // large cpu end
            while (cpu_begin <= cpu_end) {
                cpu_set_list.insert(cpu_begin);
                cpu_begin++;
            }
        } else {
            return 6; // wrong range string
        }
    }

    if (cpu_set_list.empty()) return 7; // empty cpu set list

    CPU_ZERO(cpu_set);
    for (const int& cpu : cpu_set_list) {
        CPU_SET(cpu, cpu_set);
    }

    return 0;
}