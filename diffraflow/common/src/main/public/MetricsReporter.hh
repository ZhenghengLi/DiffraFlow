#ifndef __MetricsReporter_H__
#define __MetricsReporter_H__

#include "MetricsProvider.hh"
#include <vector>
#include <map>
#include <string>
#include <cpprest/http_listener.h>
#include <pplx/pplxtasks.h>
#include <log4cxx/logger.h>

using std::vector;
using std::map;
using std::string;
using web::http::experimental::listener::http_listener;
using web::http::http_request;

namespace diffraflow {
    class MetricsReporter {
    public:
        MetricsReporter();
        ~MetricsReporter();

        bool start_msg_producer(const char* broker_address, const char* topic);
        void stop_msg_producer();

        bool start_http_server(const char* host, int port);
        void stop_http_server();

        void add(string name, MetricsProvider* mp_obj);
        void add(string name, vector<MetricsProvider*>& mp_obj_vec);
        void add(string name, MetricsProvider** mp_obj_arr, size_t len);
        void clear();

    private:
        json::value aggregate_metrics_();
        void handleGet_(http_request message);

    private:
        map<string, MetricsProvider*> metrics_scalar_;
        map<string, vector<MetricsProvider*> > metrics_array_;

        http_listener listener_;

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif