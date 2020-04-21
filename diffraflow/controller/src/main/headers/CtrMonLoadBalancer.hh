#ifndef __CtrMonLoadBalancer_H__
#define __CtrMonLoadBalancer_H__

#include <atomic>
#include <mutex>
#include <vector>
#include <condition_variable>
#include <cpprest/http_listener.h>
#include <cpprest/http_client.h>
#include <pplx/pplxtasks.h>
#include <log4cxx/logger.h>

using std::string;
using web::http::http_request;
using web::http::http_response;
using web::http::client::http_client;
using web::http::client::http_client_config;
using std::atomic;
using std::mutex;
using std::vector;

namespace diffraflow {
    class CtrMonLoadBalancer {
    public:
        CtrMonLoadBalancer();
        ~CtrMonLoadBalancer();

        bool create_monitor_clients(const char* filename, int timeout = 5000);
        bool do_one_request(http_response& response, string event_time_string);

    private:
        vector<http_client> monitor_clients_vec_;
        size_t current_index_;

        mutex mtx_client_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif
