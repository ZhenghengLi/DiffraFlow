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

        bool start_msg_producer(const char* broker_address, const char* topic);
        bool start_http_server(const char* host, int port);

        void add(string name, MetricsProvider* mp_obj);
        void add(string name, vector<MetricsProvider*>& mp_obj_vec);
        void add(string name, MetricsProvider** mp_obj_arr, size_t len);
        void clear();

    private:
        json::value aggregate_metrics_();

    private:
        map<string, MetricsProvider*> metrics_scalar_;
        map<string, vector<MetricsProvider*> > metrics_array_;


    };
}

#endif