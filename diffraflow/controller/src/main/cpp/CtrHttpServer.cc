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

bool diffraflow::CtrHttpServer::start(string host, int port) {
    if (server_status_ == kRunning) {
        LOG4CXX_WARN(logger_, "http server has already been started.");
        return false;
    }
    uri_builder uri_b;
    uri_b.set_scheme("http");
    uri_b.set_host(host);
    uri_b.set_port(port);
    listener_ = new http_listener(uri_b.to_uri());
    listener_->support(methods::GET, std::bind(&CtrHttpServer::handleGet_, this, std::placeholders::_1));
    listener_->support(methods::POST, std::bind(&CtrHttpServer::handlePost_, this, std::placeholders::_1));
    listener_->support(methods::PUT, std::bind(&CtrHttpServer::handlePut_, this, std::placeholders::_1));
    listener_->support(methods::PATCH, std::bind(&CtrHttpServer::handlePatch_, this, std::placeholders::_1));
    listener_->support(methods::DEL, std::bind(&CtrHttpServer::handleDel_, this, std::placeholders::_1));

    try {
        listener_->open().wait();
        server_status_ = kRunning;
        return true;
    } catch (std::exception& e) {
        LOG4CXX_ERROR(logger_, "failed to start http server: " << e.what());
        return false;
    } catch(...) {
        LOG4CXX_ERROR(logger_, "failed to start http server with unknown error.");
        return false;
    }
    return true;
}

void diffraflow::CtrHttpServer::stop() {
    if (listener_ == nullptr) return;

    try {
        listener_->close().wait();
    } catch (std::exception& e) {
        LOG4CXX_WARN(logger_, "exception found when closing http listener: " << e.what());
    } catch (...) {
        LOG4CXX_WARN(logger_, "an unknown exception found when closing http listener.");
    }

    delete listener_;
    listener_ = nullptr;

    server_status_ = kStopped;
    cv_status_.notify_all();

}

void diffraflow::CtrHttpServer::wait() {
    unique_lock<mutex> ulk(mtx_status_);
    cv_status_.wait(ulk,
        [this]() {return server_status_ != kRunning;}
    );
}

void diffraflow::CtrHttpServer::handleGet_(http_request message) {

}

void diffraflow::CtrHttpServer::handlePost_(http_request message) {

}

void diffraflow::CtrHttpServer::handlePut_(http_request message) {

}

void diffraflow::CtrHttpServer::handlePatch_(http_request message) {

}

void diffraflow::CtrHttpServer::handleDel_(http_request message) {

}
