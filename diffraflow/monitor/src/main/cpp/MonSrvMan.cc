#include "MonSrvMan.hh"
#include "MonConfig.hh"
#include "MonImgHttpServer.hh"

#include <future>

using std::async;

log4cxx::LoggerPtr diffraflow::MonSrvMan::logger_
    = log4cxx::Logger::getLogger("MonSrvMan");

diffraflow::MonSrvMan::MonSrvMan(MonConfig* config, const char* ingaddr_file) {
    config_obj_ = config;
    ingester_address_file_ = ingaddr_file;
    image_http_server_ = nullptr;
    running_flag_ = false;
}

diffraflow::MonSrvMan::~MonSrvMan() {

}

void diffraflow::MonSrvMan::start_run() {
    if (running_flag_) return;

    image_http_server_ = new MonImgHttpServer(config_obj_);
    if (!image_http_server_->create_ingester_clients(
        ingester_address_file_.c_str(), config_obj_->request_timeout)) {
        LOG4CXX_ERROR(logger_, "failed to create ingester clients from file: " << ingester_address_file_);
        return;
    }

    if (image_http_server_->start(config_obj_->http_host, config_obj_->http_port)) {
        LOG4CXX_INFO(logger_, "successfully started HTTP server listening "
            << config_obj_->http_host << ":" << config_obj_->http_port);
    } else {
        LOG4CXX_ERROR(logger_, "failed to start HTTP server.");
        return;
    }

    running_flag_ = true;

    // then wait for finishing
    async([this]() {
        image_http_server_->wait();
    }).wait();

}

void diffraflow::MonSrvMan::terminate() {
    if (!running_flag_) return;

    image_http_server_->stop();
    delete image_http_server_;
    image_http_server_ = nullptr;

    running_flag_ = false;
}
