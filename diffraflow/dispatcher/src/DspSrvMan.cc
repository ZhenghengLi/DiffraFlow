#include "DspSrvMan.hh"
#include "DspConfig.hh"
#include "DspSender.hh"
#include "DspImgFrmSrv.hh"
#include "DspImgFrmRecv.hh"

#include <fstream>
#include <string>

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <boost/algorithm/string.hpp>

using std::ifstream;
using std::make_pair;
using std::lock_guard;

log4cxx::LoggerPtr diffraflow::DspSrvMan::logger_ = log4cxx::Logger::getLogger("DspSrvMan");

diffraflow::DspSrvMan::DspSrvMan(DspConfig* config, const char* cmbaddr_file) {
    config_obj_ = config;
    combiner_address_file_ = cmbaddr_file;
    sender_arr_ = nullptr;
    sender_cnt_ = 0;
    imgfrm_srv_ = nullptr;
    imgfrm_recv_ = nullptr;
    running_flag_ = false;
}

diffraflow::DspSrvMan::~DspSrvMan() {}

void diffraflow::DspSrvMan::start_run() {
    if (running_flag_) return;
    // create senders
    if (create_senders_(combiner_address_file_.c_str(), config_obj_->dispatcher_id, config_obj_->max_queue_size,
            &config_obj_->other_cpu_set)) {
        LOG4CXX_INFO(logger_, sender_cnt_ << " senders are created.");
    } else {
        LOG4CXX_ERROR(logger_, "Failed to create senders.");
        return;
    }
    // create receiving server
    // TCP receiver
    imgfrm_srv_ = new DspImgFrmSrv(config_obj_->listen_host, config_obj_->listen_port, sender_arr_, sender_cnt_);
    imgfrm_srv_->set_conn_cpuset(&config_obj_->other_cpu_set);
    // UDP receiver
    imgfrm_recv_ = new DspImgFrmRecv(config_obj_->listen_host, config_obj_->listen_port, sender_arr_, sender_cnt_,
        config_obj_->dgram_recv_buffer_size, config_obj_->dgram_queue_size);

    // multiple servers start from here
    // TCP receiver
    if (imgfrm_srv_->start()) {
        LOG4CXX_INFO(logger_, "successfully started image frame TCP receiver.");
    } else {
        LOG4CXX_ERROR(logger_, "failed to start image frame TCP receiver.");
        return;
    }
    // UDP receiver
    imgfrm_recv_->start_checker(&config_obj_->other_cpu_set);
    if (imgfrm_recv_->start(config_obj_->dgram_recv_cpu_id)) {
        LOG4CXX_INFO(logger_, "successfully started image frame UDP receiver.");
    } else {
        LOG4CXX_ERROR(logger_, "failed to start image frame UDP receiver.");
        return;
    }

    // start metrics reporter
    metrics_reporter_.add("configuration", config_obj_);
    metrics_reporter_.add("image_frame_tcp_receiver", imgfrm_srv_);
    metrics_reporter_.add("image_frame_udp_receiver", imgfrm_recv_);
    metrics_reporter_.add("image_frame_senders", (MetricsProvider**)sender_arr_, sender_cnt_);
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
    async(std::launch::async, [this]() {
        lock_guard<mutex> lg(delete_mtx_);
        imgfrm_srv_->wait();
        imgfrm_recv_->wait();
    }).wait();
}

void diffraflow::DspSrvMan::terminate() {
    if (!running_flag_) return;

    // stop metrics reporter
    metrics_reporter_.stop_http_server();
    metrics_reporter_.stop_msg_producer();
    metrics_reporter_.clear();

    // stop TCP receiver
    int result = imgfrm_srv_->stop_and_close();
    if (result == 0) {
        LOG4CXX_INFO(logger_, "image frame TCP receiver is normally terminated.");
    } else if (result > 0) {
        LOG4CXX_WARN(logger_, "image frame TCP receiver is abnormally terminated with error code: " << result);
    } else {
        LOG4CXX_WARN(logger_, "image frame TCP receiver has not yet been started or already been closed.");
    }

    // stop UDP receiver
    result = imgfrm_recv_->stop_and_close();
    if (result == 0) {
        LOG4CXX_INFO(logger_, "image frame UDP receiver is normally terminated.");
    } else if (result > 0) {
        LOG4CXX_WARN(logger_, "image frame UDP receiver is abnormally terminated with error code: " << result);
    } else {
        LOG4CXX_WARN(logger_, "image frame UDP receiver has not yet been started or already been closed.");
    }
    imgfrm_recv_->stop_checker();

    lock_guard<mutex> lg(delete_mtx_);

    delete imgfrm_srv_;
    imgfrm_srv_ = nullptr;

    delete imgfrm_recv_;
    imgfrm_recv_ = nullptr;

    // delete senders
    delete_senders_();

    running_flag_ = false;
}

bool diffraflow::DspSrvMan::create_senders_(
    const char* address_list_fn, int dispatcher_id, int max_queue_size, cpu_set_t* cpuset) {
    // note: do this before staring DspImgFrmSrv
    vector<pair<string, int>> addr_vec;
    if (!read_address_list_(address_list_fn, addr_vec)) {
        LOG4CXX_ERROR(logger_, "Failed to read combiner address list.");
        return false;
    }
    sender_cnt_ = addr_vec.size();
    sender_arr_ = new DspSender*[sender_cnt_];
    for (size_t i = 0; i < addr_vec.size(); i++) {
        sender_arr_[i] = new DspSender(addr_vec[i].first, addr_vec[i].second, dispatcher_id, max_queue_size);
        if (sender_arr_[i]->start(cpuset)) {
            LOG4CXX_INFO(logger_, "successfully started sender[" << i << "]");
        } else {
            LOG4CXX_WARN(logger_, "failed to start sender[" << i << "]");
            return false;
        }
        // sender_arr_[i]->start();
    }
    return true;
}

void diffraflow::DspSrvMan::delete_senders_() {
    // note: stop DspImgFrmSrv before doing this,
    if (sender_arr_ != nullptr) {
        for (size_t i = 0; i < sender_cnt_; i++) {
            if (sender_arr_[i] != nullptr) {
                sender_arr_[i]->stop();
                delete sender_arr_[i];
                sender_arr_[i] = nullptr;
            }
        }
        delete[] sender_arr_;
        sender_arr_ = nullptr;
    }
}

bool diffraflow::DspSrvMan::read_address_list_(const char* filename, vector<pair<string, int>>& addr_vec) {
    addr_vec.clear();
    ifstream addr_file;
    addr_file.open(filename);
    if (!addr_file.is_open()) {
        LOG4CXX_ERROR(logger_, "address file open failed.");
        return false;
    }
    string oneline;
    while (true) {
        oneline = "";
        getline(addr_file, oneline);
        if (addr_file.eof()) break;
        // skip comments
        boost::trim(oneline);
        if (oneline.length() == 0) continue;
        if (oneline[0] == '#') continue;
        // extract host and port
        vector<string> host_port;
        boost::split(host_port, oneline, boost::is_any_of(":"));
        if (host_port.size() < 2) {
            LOG4CXX_ERROR(logger_, "found unknown address: " << oneline);
            return false;
        }
        boost::trim(host_port[0]);
        string host_str = host_port[0];
        boost::trim(host_port[1]);
        int port_num = atoi(host_port[1].c_str());
        if (port_num <= 0) {
            LOG4CXX_ERROR(logger_, "found unknown address: " << oneline);
            addr_file.close();
            return false;
        }
        addr_vec.push_back(make_pair(host_str, port_num));
    }
    addr_file.close();
    if (addr_vec.size() > 0) {
        return true;
    } else {
        LOG4CXX_ERROR(logger_, "empty address file: " << filename);
        return false;
    }
}
