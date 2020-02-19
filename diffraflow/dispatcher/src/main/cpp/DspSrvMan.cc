#include "DspSrvMan.hh"
#include "DspConfig.hh"
#include "DspSender.hh"
#include "DspImgFrmSrv.hh"

#include <fstream>

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <boost/algorithm/string.hpp>

using std::ifstream;
using std::make_pair;

log4cxx::LoggerPtr diffraflow::DspSrvMan::logger_
    = log4cxx::Logger::getLogger("DspSrvMan");

diffraflow::DspSrvMan::DspSrvMan(DspConfig* config, const char* cmbaddr_file) {
    config_obj_ = config;
    combiner_address_file_ = cmbaddr_file;
    sender_arr_ = nullptr;
    sender_cnt_ = 0;
    imgfrm_srv_ = nullptr;
    running_flag_ = false;
}

diffraflow::DspSrvMan::~DspSrvMan() {

}

void diffraflow::DspSrvMan::start_run() {
    if (running_flag_) return;
    // create senders
    if (create_senders_(combiner_address_file_.c_str(),
        config_obj_->dispatcher_id, config_obj_->compress_flag)) {
        LOG4CXX_INFO(logger_, sender_cnt_ << " senders are created.");
    } else {
        LOG4CXX_ERROR(logger_, "Failed to create senders.");
        return;
    }
    // start senders
    for (size_t i = 0; i < sender_cnt_; i++) sender_arr_[i]->start();
    // create receiving server
    imgfrm_srv_ = new DspImgFrmSrv(config_obj_->listen_host,
        config_obj_->listen_port, sender_arr_, sender_cnt_);
    // start to serve and block
    running_flag_ = true;
    imgfrm_srv_->serve();
}

void diffraflow::DspSrvMan::terminate() {
    if (!running_flag_) return;
    running_flag_ = false;
    // stop senders
    if (sender_arr_ == nullptr) return;
    for (size_t i = 0; i < sender_cnt_; i++) sender_arr_[i]->stop();
    // stop and delete receiving server
    imgfrm_srv_->stop();
    delete imgfrm_srv_;
    imgfrm_srv_ = nullptr;
    // delete senders
    delete_senders_();
}

bool diffraflow::DspSrvMan::create_senders_(const char* address_list_fn, int dispatcher_id, bool compress_flag) {
    // note: do this before staring DspImgFrmSrv
    vector< pair<string, int> > addr_vec;
    if (!read_address_list_(address_list_fn, addr_vec)) {
        LOG4CXX_ERROR(logger_, "Failed to read combiner address list.");
        return false;
    }
    sender_cnt_ = addr_vec.size();
    sender_arr_ = new DspSender*[sender_cnt_];
    for (size_t i = 0; i < addr_vec.size(); i++) {
        sender_arr_[i] = new DspSender(addr_vec[i].first, addr_vec[i].second, dispatcher_id, compress_flag);
        if (sender_arr_[i]->connect_to_server()) {
            LOG4CXX_INFO(logger_, "Successfully connected to combiner "
                << addr_vec[i].first.c_str() << ":" << addr_vec[i].second);
        } else {
            LOG4CXX_WARN(logger_, "Failed to do the first connection to combiner "
                << addr_vec[i].first.c_str() << ":" << addr_vec[i].second);
        }
        // sender_arr_[i]->start();
    }
    return true;
}

void diffraflow::DspSrvMan::delete_senders_() {
    // note: stop DspImgFrmSrv before doing this,
    if (sender_arr_ != nullptr) {
        for (size_t i = 0; i < sender_cnt_; i++) {
            // sender_arr_[i]->stop();
            sender_arr_[i]->send_remaining();
            delete sender_arr_[i];
        }
        delete [] sender_arr_;
    }
}

bool diffraflow::DspSrvMan::read_address_list_(const char* filename, vector< pair<string, int> >& addr_vec) {
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
        int    port_num = atoi(host_port[1].c_str());
        if (port_num <= 0) {
            LOG4CXX_ERROR(logger_, "found unknown address: " << oneline);
            return false;
        }
        addr_vec.push_back(make_pair(host_str, port_num));
    }
    if (addr_vec.size() > 0) {
        return true;
    } else {
            LOG4CXX_ERROR(logger_, "empty address file: " << filename);
        return false;
    }
}
