#ifndef __AggHttpServer_H__
#define __AggHttpServer_H__

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cpprest/http_listener.h>
#include <cpprest/http_client.h>
#include <pplx/pplxtasks.h>
#include <log4cxx/logger.h>

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

    class AggMetrics;

    class AggHttpServer {
    public:
        explicit AggHttpServer(AggMetrics* metrics);
        ~AggHttpServer();

        bool start(string host, int port);
        void stop();
        void wait();

    public:
        enum WorkerStatus { kNotStart, kRunning, kStopped };

    private:
        http_listener* listener_;

        atomic<WorkerStatus> server_status_;
        mutex mtx_status_;
        condition_variable cv_status_;

        AggMetrics* aggregated_metrics_;

    private:
        void handleGet_(http_request message);

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif