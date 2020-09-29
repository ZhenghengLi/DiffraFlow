#ifndef SndSrvMan_H
#define SndSrvMan_H

#include <vector>
#include <algorithm>
#include <string>
#include <atomic>
#include <mutex>
#include <log4cxx/logger.h>

#include "MetricsReporter.hh"

using std::atomic_bool;
using std::mutex;

namespace diffraflow {

    class SndConfig;
    class SndDatTran;
    class SndTrgSrv;

    class SndSrvMan {
    public:
        explicit SndSrvMan(SndConfig* config);
        ~SndSrvMan();

        void start_run();
        void terminate();

    private:
        SndConfig* config_obj_;
        SndDatTran* data_transfer_;

        SndTrgSrv* trigger_srv_;
        atomic_bool running_flag_;
        mutex delete_mtx_;

        MetricsReporter metrics_reporter_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
