#include "DspSrvMan.hh"
#include "DspConfig.hh"
#include "DspSender.hh"

#include <fstream>

#include <boost/log/trivial.hpp>
#include <boost/algorithm/string.hpp>

using std::ifstream;
using std::make_pair;

diffraflow::DspSrvMan::DspSrvMan(DspConfig* config) {
    config_obj_ = config;
    sender_arr_ = nullptr;
    sender_cnt_ = 0;
}

diffraflow::DspSrvMan::~DspSrvMan() {

}

bool diffraflow::DspSrvMan::create_senders(const char* address_list_fn) {
    // note: do this before staring DspImgFrmSrv
    vector< pair<string, int> > addr_vec;
    if (!read_address_list_(address_list_fn, addr_vec)) {
        BOOST_LOG_TRIVIAL(error) << "Failed to read combiner address list.";
        return false;
    }
    sender_cnt_ = addr_vec.size();
    sender_arr_ = new DspSender*[sender_cnt_];
    for (size_t i = 0; i < addr_vec.size(); i++) {
        sender_arr_[i] = new DspSender(addr_vec[i].first, addr_vec[i].second, config_obj_->dispatcher_id);
        if (!sender_arr_[i]->connect_to_combiner()) {
            BOOST_LOG_TRIVIAL(warning) << sprintf("Failed to do the first connection to combiner %s:%d",
                addr_vec[i].first.c_str(), addr_vec[i].second);
        }
        sender_arr_[i]->start();
    }
    return true;
}

void diffraflow::DspSrvMan::delete_senders() {
    // note: stop DspImgFrmSrv before doing this,
    if (sender_arr_ != nullptr) {
        for (size_t i = 0; i < sender_cnt_; i++) {
            sender_arr_[i]->stop();
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
        BOOST_LOG_TRIVIAL(error) << "address file open failed.";
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
            BOOST_LOG_TRIVIAL(error) << "found unknown address: " << oneline;
            return false;
        }
        boost::trim(host_port[0]);
        string host_str = host_port[0];
        boost::trim(host_port[1]);
        int    port_num = atoi(host_port[1].c_str());
        if (port_num <= 0) {
            BOOST_LOG_TRIVIAL(error) << "found unknown address: " << oneline;
            return false;
        }
        addr_vec.push_back(make_pair(host_str, port_num));
    }
    if (addr_vec.size() > 0) {
        return true;
    } else {
            BOOST_LOG_TRIVIAL(error) << "empty address file: " << filename;
        return false;
    }
}
