#ifndef MonSrvMan_H
#define MonSrvMan_H

#include <atomic>
#include <mutex>
#include <log4cxx/logger.h>

#include "MetricsReporter.hh"

using std::atomic_bool;
using std::string;
using std::mutex;

namespace diffraflow {

    class MonConfig;
    class MonImgHttpServer;

    class MonSrvMan {
    public:
        MonSrvMan(MonConfig* config, const char* ingaddr_file);
        ~MonSrvMan();

        void start_run();
        void terminate();

    private:
        MonConfig* config_obj_;
        MonImgHttpServer* image_http_server_;

        string ingester_address_file_;

        atomic_bool running_flag_;
        mutex delete_mtx_;

        // metrics
        MetricsReporter metrics_reporter_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
