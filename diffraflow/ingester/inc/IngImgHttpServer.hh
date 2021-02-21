#ifndef __IngImgHttpServer_H__
#define __IngImgHttpServer_H__

#include <string>
#include <atomic>
#include <cpprest/http_listener.h>
#include <pplx/pplxtasks.h>
#include <log4cxx/logger.h>

#include "MetricsProvider.hh"

using std::string;
using web::http::experimental::listener::http_listener;
using web::http::http_request;
using std::atomic;

namespace diffraflow {

    class IngImageFilter;

    class IngImgHttpServer : public MetricsProvider {
    public:
        IngImgHttpServer(IngImageFilter* img_filter, int ing_id);
        ~IngImgHttpServer();

        bool start(string host, int port);
        void stop();

    public:
        struct {
            atomic<uint64_t> total_request_counts;
            atomic<uint64_t> total_sent_counts;
        } metrics;

        json::value collect_metrics() override;

    private:
        void handleGet_(http_request message);

    private:
        IngImageFilter* image_filter_;
        http_listener* listener_;

        int ingester_id_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
