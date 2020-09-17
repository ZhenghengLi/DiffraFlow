#include "SndSrvMan.hh"
#include "SndConfig.hh"
#include "SndDatTran.hh"
#include "SndTrgSrv.hh"

#include <fstream>
#include <string>

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <boost/algorithm/string.hpp>

using std::ifstream;
using std::make_pair;

log4cxx::LoggerPtr diffraflow::SndSrvMan::logger_ = log4cxx::Logger::getLogger("SndSrvMan");

diffraflow::SndSrvMan::SndSrvMan(SndConfig* config) {
    config_obj_ = config;
    data_transfer_ = nullptr;
    trigger_srv_ = nullptr;
    running_flag_ = false;
}

diffraflow::SndSrvMan::~SndSrvMan() {}

void diffraflow::SndSrvMan::start_run() {
    if (running_flag_) return;

    // create data transfer
    data_transfer_ = new SndDatTran(config_obj_);
    // create trigger server
    trigger_srv_ = new SndTrgSrv(config_obj_->listen_host, config_obj_->listen_port, data_transfer_);

    // prepare data transfer
    if (config_obj_->sender_type == "TCP") {
        if (data_transfer_->create_tcp_sender(
                config_obj_->dispatcher_host, config_obj_->dispatcher_port, config_obj_->sender_id)) {
            LOG4CXX_INFO(logger_, "successfully created TCP sender for dispatcher "
                                      << config_obj_->dispatcher_host << ":" << config_obj_->dispatcher_port);
        } else {
            LOG4CXX_ERROR(logger_, "failed to create TCP sender for dispatcher " << config_obj_->dispatcher_host << ":"
                                                                                 << config_obj_->dispatcher_port);
            return;
        }
    } else if (config_obj_->sender_type == "UDP") {
        if (data_transfer_->create_udp_sender(config_obj_->dispatcher_host, config_obj_->dispatcher_port)) {
            LOG4CXX_INFO(logger_, "successfully created UDP sender for dispatcher "
                                      << config_obj_->dispatcher_host << ":" << config_obj_->dispatcher_port);
        } else {
            LOG4CXX_ERROR(logger_, "failed to create UDP sender for dispatcher " << config_obj_->dispatcher_host << ":"
                                                                                 << config_obj_->dispatcher_port);
            return;
        }
    } else {
        LOG4CXX_ERROR(logger_, "wrong sender type: " << config_obj_->sender_type);
        return;
    }
    // start sender thread and bind cpu
    if (data_transfer_->start_sender(config_obj_->sender_cpu_id)) {
        LOG4CXX_INFO(logger_, "successfully started sender thread with cpu " << config_obj_->sender_cpu_id);
    } else {
        LOG4CXX_ERROR(logger_, "failed to start sender thread with cpu " << config_obj_->sender_cpu_id);
        return;
    }

    // start trigger server
    if (trigger_srv_->start()) {
        LOG4CXX_INFO(logger_, "successfully started trigger server.")
    } else {
        LOG4CXX_ERROR(logger_, "failed to start trigger server.")
        return;
    }

    // start metrics reporter
    metrics_reporter_.add("configuration", config_obj_);
    metrics_reporter_.add("data_transfer", data_transfer_);
    metrics_reporter_.add("trigger_server", trigger_srv_);
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

    // wait for finishing
    async([this]() { trigger_srv_->wait(); }).wait();
}

void diffraflow::SndSrvMan::terminate() {
    if (!running_flag_) return;

    // stop metrics reporter
    metrics_reporter_.stop_http_server();
    metrics_reporter_.stop_msg_producer();
    metrics_reporter_.clear();

    // stop and delete trigger server
    int result = trigger_srv_->stop_and_close();
    if (result == 0) {
        LOG4CXX_INFO(logger_, "trigger server is normally terminated.");
    } else if (result > 0) {
        LOG4CXX_WARN(logger_, "trigger server is abnormally terminated with error code: " << result);
    } else {
        LOG4CXX_WARN(logger_, "trigger server has not yet been started or already been closed.");
    }
    delete trigger_srv_;
    trigger_srv_ = nullptr;

    data_transfer_->stop_sender();
    data_transfer_->delete_sender();
    delete data_transfer_;
    data_transfer_ = nullptr;

    running_flag_ = false;
}
