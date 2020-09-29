#ifndef __AggSrvMan_H__
#define __AggSrvMan_H__

#include <atomic>
#include <mutex>
#include <log4cxx/logger.h>

#include "MetricsReporter.hh"

using std::atomic_bool;
using std::string;
using std::mutex;

namespace diffraflow {

    class AggConfig;
    class AggMetrics;
    class AggHttpServer;

    class AggSrvMan {
    public:
        explicit AggSrvMan(AggConfig* config);
        ~AggSrvMan();

        void start_run();
        void terminate();

    private:
        AggConfig* config_obj_;

        AggMetrics* aggregated_metrics_;
        AggHttpServer* http_server_;

        mutex delete_mtx_;

        atomic_bool running_flag_;

        MetricsReporter metrics_reporter_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif