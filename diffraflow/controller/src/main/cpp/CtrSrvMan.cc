#include "CtrSrvMan.hh"
#include "CtrConfig.hh"
#include "DynamicConfiguration.hh"
#include "CtrMonLoadBalancer.hh"
#include "CtrHttpServer.hh"

#include <future>

using std::async;

log4cxx::LoggerPtr diffraflow::CtrSrvMan::logger_ = log4cxx::Logger::getLogger("CtrSrvMan");

diffraflow::CtrSrvMan::CtrSrvMan(CtrConfig* config, const char* monaddr_file, DynamicConfiguration* zk_conf_client) {
    config_obj_ = config;
    if (monaddr_file != nullptr) {
        monitor_address_file_ = monaddr_file;
    }
    zookeeper_config_client_ = zk_conf_client;
    monitor_load_balancer_ = nullptr;
    http_server_ = nullptr;
    running_flag_ = false;
}

diffraflow::CtrSrvMan::~CtrSrvMan() {}

void diffraflow::CtrSrvMan::start_run() {
    if (running_flag_) return;

    if (monitor_address_file_.empty() && zookeeper_config_client_ == nullptr) {
        LOG4CXX_ERROR(logger_, "nothing to start.");
        return;
    }

    if (!monitor_address_file_.empty()) {
        monitor_load_balancer_ = new CtrMonLoadBalancer();

        if (monitor_load_balancer_->create_monitor_clients(
                monitor_address_file_.c_str(), config_obj_->request_timeout)) {
            LOG4CXX_INFO(logger_, "successfully created monitor load balancer.");
        } else {
            LOG4CXX_ERROR(logger_, "failed to create monitor load balancer.");
            return;
        }
    }

    http_server_ = new CtrHttpServer(monitor_load_balancer_, zookeeper_config_client_);
    if (http_server_->start(config_obj_->http_host, config_obj_->http_port)) {
        LOG4CXX_INFO(logger_,
            "successfully started HTTP server listening " << config_obj_->http_host << ":" << config_obj_->http_port);
    } else {
        LOG4CXX_ERROR(logger_, "failed to start HTTP server.");
        return;
    }

    running_flag_ = true;

    // then wait for finishing
    async(std::launch::async, [this]() { http_server_->wait(); }).wait();
}

void diffraflow::CtrSrvMan::terminate() {
    if (!running_flag_) return;

    if (http_server_ != nullptr) {
        http_server_->stop();
        delete http_server_;
        http_server_ = nullptr;
    }

    if (monitor_load_balancer_ != nullptr) {
        delete monitor_load_balancer_;
        monitor_load_balancer_ = nullptr;
    }

    running_flag_ = false;
}
