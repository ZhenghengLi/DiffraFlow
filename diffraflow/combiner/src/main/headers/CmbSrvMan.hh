#ifndef CmbSrvMan_H
#define CmbSrvMan_H

#include <atomic>
#include <log4cxx/logger.h>

#include "MetricsReporter.hh"

using std::atomic_bool;

namespace diffraflow {

    class CmbConfig;
    class CmbImgCache;
    class CmbImgFrmSrv;
    class CmbImgDatSrv;

    class CmbSrvMan {
    public:
        explicit CmbSrvMan(CmbConfig* config);
        ~CmbSrvMan();

        void start_run();
        void terminate();

    private:
        CmbConfig* config_obj_;
        CmbImgCache* image_cache_;

        CmbImgFrmSrv* imgfrm_srv_;
        CmbImgDatSrv* imgdat_srv_;
        atomic_bool running_flag_;

        MetricsReporter metrics_reporter_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif
