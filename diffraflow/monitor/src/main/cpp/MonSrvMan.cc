#include "MonSrvMan.hh"
#include "MonConfig.hh"
#include "MonImgHttpServer.hh"

#include <future>

using std::async;

log4cxx::LoggerPtr diffraflow::MonSrvMan::logger_ = log4cxx::Logger::getLogger("MonSrvMan");

diffraflow::MonSrvMan::MonSrvMan(MonConfig* config, const char* ingaddr_file) {
    config_obj_ = config;
    ingester_address_file_ = ingaddr_file;
    image_http_server_ = nullptr;
    running_flag_ = false;
}

diffraflow::MonSrvMan::~MonSrvMan() {}

void diffraflow::MonSrvMan::start_run() {
    if (running_flag_) return;

    image_http_server_ = new MonImgHttpServer(config_obj_);
    if (!image_http_server_->create_ingester_clients(ingester_address_file_.c_str(), config_obj_->request_timeout)) {
        LOG4CXX_ERROR(logger_, "failed to create ingester clients from file: " << ingester_address_file_);
        return;
    }

    if (image_http_server_->start(config_obj_->image_http_host, config_obj_->image_http_port)) {
        LOG4CXX_INFO(logger_, "successfully started HTTP server listening " << config_obj_->image_http_host << ":"
                                                                            << config_obj_->image_http_port);
    } else {
        LOG4CXX_ERROR(logger_, "failed to start HTTP server.");
        return;
    }

    // start metrics reporter
    metrics_reporter_.add("configuration", config_obj_);
    metrics_reporter_.add("image_http_server", image_http_server_);
    if (config_obj_->metrics_pulsar_params_are_set()) {
        if (metrics_reporter_.start_msg_producer(config_obj_->metrics_pulsar_broker_address,
                config_obj_->metrics_pulsar_topic_name, config_obj_->metrics_pulsar_message_key,
                config_obj_->metrics_pulsar_report_period)) {
            LOG4CXX_INFO(logger_, "Successfully started pulsar producer to periodically report metrics.");
        } else {
            LOG4CXX_ERROR(logger_, "Failed to start pulsar producer to periodically report metrics.");
            return;
        }
    }
    if (config_obj_->metrics_http_params_are_set()) {
        if (metrics_reporter_.start_http_server(config_obj_->metrics_http_host, config_obj_->metrics_http_port)) {
            LOG4CXX_INFO(logger_, "Successfully started http server for metrics service.");
        } else {
            LOG4CXX_ERROR(logger_, "Failed to start http server for metrics service.");
        }
    }

    running_flag_ = true;

    // then wait for finishing
    async([this]() { image_http_server_->wait(); }).wait();
}

void diffraflow::MonSrvMan::terminate() {
    if (!running_flag_) return;

    // stop metrics reporter
    metrics_reporter_.stop_http_server();
    metrics_reporter_.stop_msg_producer();
    metrics_reporter_.clear();

    image_http_server_->stop();
    delete image_http_server_;
    image_http_server_ = nullptr;

    running_flag_ = false;
}
