#include "AggSrvMan.hh"
#include "AggConfig.hh"
#include "AggMetrics.hh"
#include "AggHttpServer.hh"

#include <future>

using std::async;
using std::lock_guard;

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

    aggregated_metrics_ =
        new AggMetrics(config_obj_->pulsar_url, config_obj_->subscription_name, config_obj_->read_compacted, 8, 6);
    if (!config_obj_->controller_topic.empty()) {
        if (aggregated_metrics_->start_consumer("controller", config_obj_->controller_topic)) {
            LOG4CXX_INFO(logger_,
                "successfully started metrics consumer for controller topic " << config_obj_->controller_topic);
        } else {
            LOG4CXX_ERROR(
                logger_, "failed to start metrics consumer for controller topic " << config_obj_->controller_topic);
            return;
        }
    }
    if (!config_obj_->sender_topic.empty()) {
        if (aggregated_metrics_->start_consumer("sender", config_obj_->sender_topic)) {
            LOG4CXX_INFO(
                logger_, "successfully started metrics consumer for sender topic " << config_obj_->sender_topic);
        } else {
            LOG4CXX_ERROR(logger_, "failed to start metrics consumer for sender topic " << config_obj_->sender_topic);
            return;
        }
    }
    if (!config_obj_->dispatcher_topic.empty()) {
        if (aggregated_metrics_->start_consumer("dispatcher", config_obj_->dispatcher_topic)) {
            LOG4CXX_INFO(logger_,
                "successfully started metrics consumer for dispatcher topic " << config_obj_->dispatcher_topic);
        } else {
            LOG4CXX_ERROR(
                logger_, "failed to start metrics consumer for dispatcher topic " << config_obj_->dispatcher_topic);
            return;
        }
    }
    if (!config_obj_->combiner_topic.empty()) {
        if (aggregated_metrics_->start_consumer("combiner", config_obj_->combiner_topic)) {
            LOG4CXX_INFO(
                logger_, "successfully started metrics consumer for combiner topic " << config_obj_->combiner_topic);
        } else {
            LOG4CXX_ERROR(
                logger_, "failed to start metrics consumer for combiner topic " << config_obj_->combiner_topic);
            return;
        }
    }
    if (!config_obj_->ingester_topic.empty()) {
        if (aggregated_metrics_->start_consumer("ingester", config_obj_->ingester_topic)) {
            LOG4CXX_INFO(
                logger_, "successfully started metrics consumer for ingester topic " << config_obj_->ingester_topic);
        } else {
            LOG4CXX_ERROR(
                logger_, "failed to start metrics consumer for ingester topic " << config_obj_->ingester_topic);
            return;
        }
    }
    if (!config_obj_->monitor_topic.empty()) {
        if (aggregated_metrics_->start_consumer("monitor", config_obj_->monitor_topic)) {
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

    // start metrics reporter
    metrics_reporter_.add("configuration", config_obj_);
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
            return;
        }
    }

    running_flag_ = true;

    // then wait for finishing
    async(std::launch::async, [this]() {
        lock_guard<mutex> lg(delete_mtx_);
        http_server_->wait();
        aggregated_metrics_->wait();
    }).wait();
}

void diffraflow::AggSrvMan::terminate() {
    if (!running_flag_) return;

    // stop metrics reporter
    metrics_reporter_.stop_http_server();
    metrics_reporter_.stop_msg_producer();
    metrics_reporter_.clear();

    // stop http server
    http_server_->stop();

    // stop metrics consumers
    aggregated_metrics_->stop_all();

    lock_guard<mutex> lg(delete_mtx_);

    delete http_server_;
    http_server_ = nullptr;

    delete aggregated_metrics_;
    aggregated_metrics_ = nullptr;

    running_flag_ = false;
}
