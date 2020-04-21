#include "CtrHttpServer.hh"
#include "CtrMonLoadBalancer.hh"
#include "DynamicConfiguration.hh"

using namespace web;
using namespace http;
using namespace experimental::listener;

using std::lock_guard;
using std::unique_lock;

log4cxx::LoggerPtr diffraflow::CtrHttpServer::logger_
    = log4cxx::Logger::getLogger("CtrHttpServer");

diffraflow::CtrHttpServer::CtrHttpServer(CtrMonLoadBalancer* mon_ld_bl, DynamicConfiguration* zk_conf_client) {
    listener_ = nullptr;
    server_status_ = kNotStart;
    monitor_load_balancer_ = mon_ld_bl;
    zookeeper_config_client_ = zk_conf_client;
}

diffraflow::CtrHttpServer::~CtrHttpServer() {

}
