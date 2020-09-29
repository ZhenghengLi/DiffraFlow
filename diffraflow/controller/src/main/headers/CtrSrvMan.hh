#ifndef CtrSrvMan_H
#define CtrSrvMan_H

#include <atomic>
#include <mutex>
#include <log4cxx/logger.h>

using std::atomic_bool;
using std::string;
using std::mutex;

namespace diffraflow {

    class CtrConfig;
    class DynamicConfiguration;
    class CtrMonLoadBalancer;
    class CtrHttpServer;

    class CtrSrvMan {
    public:
        CtrSrvMan(CtrConfig* config, const char* monaddr_file, DynamicConfiguration* zk_conf_client);
        ~CtrSrvMan();

        void start_run();
        void terminate();

    private:
        CtrConfig* config_obj_;

        string monitor_address_file_;
        DynamicConfiguration* zookeeper_config_client_;
        CtrMonLoadBalancer* monitor_load_balancer_;
        CtrHttpServer* http_server_;

        atomic_bool running_flag_;
        mutex delete_mtx_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
