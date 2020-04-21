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

    class ImageWithFeature;
    class ImageAnalysisResult;
    class MonConfig;

    class MonImgHttpServer {
    public:
        explicit MonImgHttpServer(MonConfig* conf_obj);
        ~MonImgHttpServer();

        bool create_ingester_clients(const char* filename, int timeout = 1000);

        bool start(string host, int port);
        void stop();
        void wait();

    public:
        enum WorkerStatus {kNotStart, kRunning, kStopped};

    private:
        bool request_one_image_(const string event_time_string,
            ImageWithFeature& image_with_feature, string& ingester_id_str);
        void do_analysis_(const ImageWithFeature& image_with_feature,
            ImageAnalysisResult& image_analysis_result);

    private:
        http_listener*  listener_;

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
}

#endif