#ifndef MonSrvMan_H
#define MonSrvMan_H

#include <atomic>
#include <log4cxx/logger.h>

using std::atomic_bool;

namespace diffraflow {

    class MonConfig;

    class MonSrvMan {
    public:
        explicit MonSrvMan(MonConfig* config);
        ~MonSrvMan();

        void start_run();
        void terminate();

    private:
        MonConfig* config_obj_;

        atomic_bool running_flag_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif
