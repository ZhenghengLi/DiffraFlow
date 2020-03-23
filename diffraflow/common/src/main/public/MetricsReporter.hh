#ifndef __MetricsReporter_H__
#define __MetricsReporter_H__

#include "MetricsProvider.hh"
#include <vector>
#include <map>
#include <string>

using std::vector;
using std::map;
using std::string;

namespace diffraflow {
    class MetricsReporter {
    public:
        MetricsReporter();
        ~MetricsReporter();

        void add(string name, MetricsProvider* mp_obj);
        void add(string name, vector<MetricsProvider*> mp_obj_vec);
        void add(string name, MetricsProvider** mp_obj_arr, size_t len);

    private:
        map<string, MetricsProvider*> metrics_scalar_;
        map<string, vector<MetricsProvider*> > metrics_array_;


    };
}

#endif