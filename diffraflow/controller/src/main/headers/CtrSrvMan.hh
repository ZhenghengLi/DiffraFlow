#ifndef CtrSrvMan_H
#define CtrSrvMan_H

#include <atomic>
#include <log4cxx/logger.h>

using std::atomic_bool;
using std::string;

namespace diffraflow {

    class CtrConfig;

    class CtrSrvMan {
    public:
        CtrSrvMan(CtrConfig* config, const char* monaddr_file);
        ~CtrSrvMan();

        void start_run();
        void terminate();

    private:
        CtrConfig* config_obj_;

        string monitor_address_file_;

        atomic_bool running_flag_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif
