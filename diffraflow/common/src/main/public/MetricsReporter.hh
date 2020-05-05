#ifndef __MetricsReporter_H__
#define __MetricsReporter_H__

#include "MetricsProvider.hh"
#include <vector>
#include <map>
#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include <cpprest/http_listener.h>
#include <pplx/pplxtasks.h>

#include <pulsar/Client.h>
#include <pulsar/Producer.h>
#include <pulsar/Message.h>
#include <pulsar/Result.h>
#include <pulsar/ProducerConfiguration.h>

#include <log4cxx/logger.h>

using std::vector;
using std::map;
using std::string;
using std::atomic_bool;
using web::http::experimental::listener::http_listener;
using web::http::http_request;
using std::thread;
using std::mutex;
using std::condition_variable;

namespace diffraflow {
    class MetricsReporter {
    public:
        MetricsReporter();
        ~MetricsReporter();

        bool start_msg_producer(
            string broker_address, string topic, string msg_key, size_t report_period /* milliseconds */
        );
        void stop_msg_producer();

        bool start_http_server(string host, int port);
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
        map<string, vector<MetricsProvider*>> metrics_array_;

        http_listener* listener_;
        pulsar::Client* pulsar_client_;
        pulsar::Producer* pulsar_producer_;
        string message_key_;
        size_t report_period_;
        thread* sender_thread_;
        atomic_bool sender_is_running_;
        mutex wait_mtx_;
        condition_variable wait_cv_;
        mutex read_mtx_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif