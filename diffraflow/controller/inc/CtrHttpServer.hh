#ifndef __CtrHttpServer_H__
#define __CtrHttpServer_H__

#include <atomic>
#include <mutex>
#include <vector>
#include <condition_variable>
#include <cpprest/http_listener.h>
#include <cpprest/http_client.h>
#include <pplx/pplxtasks.h>
#include <log4cxx/logger.h>

#include "MetricsProvider.hh"

using std::string;
using web::http::experimental::listener::http_listener;
using web::http::http_request;
using web::http::client::http_client;
using web::http::client::http_client_config;
using std::atomic;
using std::mutex;
using std::condition_variable;
using std::vector;

namespace diffraflow {

    class CtrMonLoadBalancer;
    class DynamicConfiguration;

    class CtrHttpServer : public MetricsProvider {
    public:
        explicit CtrHttpServer(CtrMonLoadBalancer* mon_ld_bl, DynamicConfiguration* zk_conf_client);
        ~CtrHttpServer();

        bool start(string host, int port);
        void stop();
        void wait();

    public:
        struct {
            atomic<uint64_t> total_options_request_count;
            atomic<uint64_t> total_get_request_count;
            atomic<uint64_t> total_event_request_count;
            atomic<uint64_t> total_event_sent_count;
            atomic<uint64_t> total_post_request_count;
            atomic<uint64_t> total_put_request_count;
            atomic<uint64_t> total_patch_request_count;
            atomic<uint64_t> total_delete_request_count;
        } http_metrics;

        json::value collect_metrics() override;

    public:
        enum WorkerStatus { kNotStart, kRunning, kStopped };

    private:
        http_listener* listener_;

        atomic<WorkerStatus> server_status_;
        mutex mtx_status_;
        condition_variable cv_status_;

        CtrMonLoadBalancer* monitor_load_balancer_;
        DynamicConfiguration* zookeeper_config_client_;

    private:
        void handleGet_(http_request message);     // zookeeper_fetch_config | zookeeper_get_children
        void handlePost_(http_request message);    // zookeeper_create_config
        void handlePut_(http_request message);     // zookeeper_change_config
        void handlePatch_(http_request message);   // zookeeper_fetch_config & zookeeper_change_config
        void handleDel_(http_request message);     // zookeeper_delete_config
        void handleOptions_(http_request message); // for preflight request from browser

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
