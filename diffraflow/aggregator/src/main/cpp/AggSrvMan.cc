#include "AggSrvMan.hh"
#include "AggConfig.hh"
#include "AggMetrics.hh"
#include "AggHttpServer.hh"

#include <future>

using std::async;

log4cxx::LoggerPtr diffraflow::AggSrvMan::logger_ = log4cxx::Logger::getLogger("AggSrvMan");

diffraflow::AggSrvMan::AggSrvMan(AggConfig* config) {
    config_obj_ = config;
    aggregated_metrics_ = nullptr;
    http_server_ = nullptr;
    running_flag_ = false;
}

diffraflow::AggSrvMan::~AggSrvMan() {
    //
}

void diffraflow::AggSrvMan::start_run() {
    if (running_flag_) return;

    aggregated_metrics_ = new AggMetrics(config_obj_->pulsar_url, 5);
    if (!config_obj_->sender_topic.empty()) {
        if (aggregated_metrics_->start_sender_consumer(config_obj_->sender_topic)) {
            LOG4CXX_INFO(
                logger_, "successfully started metrics consumer for sender topic " << config_obj_->sender_topic);
        } else {
            LOG4CXX_ERROR(logger_, "failed to start metrics consumer for sender topic " << config_obj_->sender_topic);
            return;
        }
    }
    if (!config_obj_->dispatcher_topic.empty()) {
        if (aggregated_metrics_->start_dispatcher_consumer(config_obj_->dispatcher_topic)) {
            LOG4CXX_INFO(logger_,
                "successfully started metrics consumer for dispatcher topic " << config_obj_->dispatcher_topic);
        } else {
            LOG4CXX_ERROR(
                logger_, "failed to start metrics consumer for dispatcher topic " << config_obj_->dispatcher_topic);
            return;
        }
    }
    if (!config_obj_->combiner_topic.empty()) {
        if (aggregated_metrics_->start_combiner_consumer(config_obj_->combiner_topic)) {
            LOG4CXX_INFO(
                logger_, "successfully started metrics consumer for combiner topic " << config_obj_->combiner_topic);
        } else {
            LOG4CXX_ERROR(
                logger_, "failed to start metrics consumer for combiner topic " << config_obj_->combiner_topic);
            return;
        }
    }
    if (!config_obj_->ingester_topic.empty()) {
        if (aggregated_metrics_->start_ingester_consumer(config_obj_->ingester_topic)) {
            LOG4CXX_INFO(
                logger_, "successfully started metrics consumer for ingester topic " << config_obj_->ingester_topic);
        } else {
            LOG4CXX_ERROR(
                logger_, "failed to start metrics consumer for ingester topic " << config_obj_->ingester_topic);
            return;
        }
    }
    if (!config_obj_->monitor_topic.empty()) {
        if (aggregated_metrics_->start_monitor_consumer(config_obj_->monitor_topic)) {
            LOG4CXX_INFO(
                logger_, "successfully started metrics consumer for monitor topic " << config_obj_->monitor_topic);
        } else {
            LOG4CXX_ERROR(logger_, "failed to start metrics consumer for monitor topic " << config_obj_->monitor_topic);
            return;
        }
    }

    http_server_ = new AggHttpServer(aggregated_metrics_);
    if (http_server_->start(config_obj_->http_server_host, config_obj_->http_server_port)) {
        LOG4CXX_INFO(logger_, "successfully started HTTP server listening " << config_obj_->http_server_host << ":"
                                                                            << config_obj_->http_server_port);
    } else {
        LOG4CXX_ERROR(logger_, "failed to start HTTP server.");
        return;
    }

    running_flag_ = true;

    // then wait for finishing
    async(std::launch::async, [this]() {
        aggregated_metrics_->wait_all();
        http_server_->wait();
    }).wait();
}

void diffraflow::AggSrvMan::terminate() {
    if (!running_flag_) return;

    if (http_server_ != nullptr) {
        http_server_->stop();
        delete http_server_;
        http_server_ = nullptr;
    }

    if (aggregated_metrics_ != nullptr) {
        aggregated_metrics_->stop_sender_consumer();
        aggregated_metrics_->stop_dispatcher_consumer();
        aggregated_metrics_->stop_combiner_consumer();
        aggregated_metrics_->stop_ingester_consumer();
        aggregated_metrics_->stop_monitor_consumer();
    }
    delete aggregated_metrics_;
    aggregated_metrics_ = nullptr;

    running_flag_ = false;
}
