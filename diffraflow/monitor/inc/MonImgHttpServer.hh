#ifndef __MonImgHttpServer_H__
#define __MonImgHttpServer_H__

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
using std::pair;

namespace diffraflow {

    struct ImageDataFeature;
    struct ImageAnalysisResult;
    class MonConfig;

    class MonImgHttpServer : public MetricsProvider {
    public:
        explicit MonImgHttpServer(MonConfig* conf_obj);
        ~MonImgHttpServer();

        bool create_ingester_clients(const char* filename, int timeout = 1000);

        bool start(string host, int port);
        void stop();
        void wait();

    public:
        struct {
            atomic<uint64_t> total_request_counts;
            atomic<uint64_t> total_sent_counts;
        } metrics;

        json::value collect_metrics() override;

    public:
        enum WorkerStatus { kNotStart, kRunning, kStopped };

    private:
        bool request_one_image_(const string key_string, ImageDataFeature& image_data_feature, string& ingester_id_str);
        void do_analysis_(const ImageDataFeature& image_data_feature, ImageAnalysisResult& image_analysis_result);

    private:
        http_listener* listener_;

        atomic<WorkerStatus> server_status_;
        mutex mtx_status_;
        condition_variable cv_status_;

        mutex mtx_client_;

        MonConfig* config_obj_;

    private:
        void handleGet_(http_request message);

    private:
        vector<http_client> ingester_clients_vec_;
        size_t current_index_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif