#ifndef __MonImgHttpServer_H__
#define __MonImgHttpServer_H__

#include <atomic>
#include <mutex>
#include <vector>
#include <condition_variable>
#include <cpprest/http_listener.h>
#include <pplx/pplxtasks.h>
#include <log4cxx/logger.h>

using std::string;
using web::http::experimental::listener::http_listener;
using web::http::http_request;
using std::atomic;
using std::mutex;
using std::condition_variable;
using std::vector;
using std::pair;

namespace diffraflow {
    class MonImgHttpServer {
    public:
        MonImgHttpServer();
        ~MonImgHttpServer();

        bool load_ingaddr_list(const char* filename);

        bool start(string host, int port);
        void stop();
        void wait();

    public:
        enum WorkerStatus {kNotStart, kRunning, kStopped};

    private:
        http_listener*  listener_;

        atomic<WorkerStatus> server_status_;
        mutex mtx_status_;
        condition_variable cv_status_;

    private:
        void handleGet_(http_request message);

    private:
        vector<string> ingester_addresses_vec_;
        size_t previous_index_;
        size_t current_index_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif