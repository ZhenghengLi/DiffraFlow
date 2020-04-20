#ifndef MonSrvMan_H
#define MonSrvMan_H

#include <atomic>
#include <log4cxx/logger.h>

using std::atomic_bool;
using std::string;

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

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif
