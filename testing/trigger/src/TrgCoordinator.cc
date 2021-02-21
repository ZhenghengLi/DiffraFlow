#include "TrgCoordinator.hh"
#include "TrgClient.hh"

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <boost/algorithm/string.hpp>

using std::ifstream;
using std::vector;
using std::string;
using std::pair;
using std::unique_lock;
using std::lock_guard;
using std::chrono::microseconds;
using std::chrono::duration;
using std::chrono::system_clock;
using std::micro;
using std::cout;
using std::endl;
using std::flush;

log4cxx::LoggerPtr diffraflow::TrgCoordinator::logger_ = log4cxx::Logger::getLogger("TrgCoordinator");

diffraflow::TrgCoordinator::TrgCoordinator() {
    trigger_client_arr_ = nullptr;
    trigger_client_cnt_ = 0;
}

diffraflow::TrgCoordinator::~TrgCoordinator() { delete_trigger_clients(); }

void diffraflow::TrgCoordinator::delete_trigger_clients() {
    if (trigger_client_arr_ != nullptr) {
        for (size_t i = 0; i < trigger_client_cnt_; i++) {
            if (trigger_client_arr_[i] != nullptr) {
                trigger_client_arr_[i]->stop();
                delete trigger_client_arr_[i];
                trigger_client_arr_[i] = nullptr;
            }
        }
        delete[] trigger_client_arr_;
        trigger_client_arr_ = nullptr;
        trigger_client_cnt_ = 0;
    }
}

bool diffraflow::TrgCoordinator::create_trigger_clients(const char* sender_list_file, uint32_t trigger_id) {
    if (trigger_client_arr_ != nullptr) {
        LOG4CXX_WARN(logger_, "triggers are already created and are running.");
        return false;
    }
    // read sender list
    vector<pair<string, int>> address_vec;
    if (!read_address_list_(sender_list_file, address_vec)) {
        LOG4CXX_ERROR(logger_, "failed to read sender list file " << sender_list_file);
        return false;
    }
    trigger_client_cnt_ = address_vec.size();
    trigger_client_arr_ = new TrgClient*[trigger_client_cnt_];
    bool all_ready_flag = true;
    for (size_t i = 0; i < trigger_client_cnt_; i++) {
        trigger_client_arr_[i] = new TrgClient(address_vec[i].first, address_vec[i].second, trigger_id);
        if (trigger_client_arr_[i]->start()) {
            LOG4CXX_INFO(logger_,
                "successfully started trigger client for sender " << trigger_client_arr_[i]->get_server_address());
        } else {
            LOG4CXX_WARN(
                logger_, "failed to start trigger client for sender " << trigger_client_arr_[i]->get_server_address());
            all_ready_flag = false;
        }
    }
    return all_ready_flag;
}

bool diffraflow::TrgCoordinator::trigger_one_event(uint32_t event_index) {
    if (trigger_client_arr_ == nullptr) {
        LOG4CXX_WARN(logger_, "sender clients are not yet created and running.");
        return false;
    }

    for (size_t i = 0; i < trigger_client_cnt_; i++) {
        trigger_client_arr_[i]->trigger(event_index);
    }
    vector<int> fail_module_ids_vec;
    for (size_t i = 0; i < trigger_client_cnt_; i++) {
        if (!trigger_client_arr_[i]->wait()) {
            fail_module_ids_vec.push_back(i);
        }
    }

    if (fail_module_ids_vec.size() > 0) {
        LOG4CXX_WARN(logger_, "event of index " << event_index << " is not fully triggered.");
        cout << "these modules failed to be triggered:";
        for (size_t i = 0; i < fail_module_ids_vec.size(); i++) {
            cout << " " << fail_module_ids_vec[i];
        }
        cout << endl;
        return false;
    } else {
        LOG4CXX_INFO(logger_, "event of index " << event_index << " is successfully triggered.");
        return true;
    }
}

bool diffraflow::TrgCoordinator::trigger_many_events(
    uint32_t start_event_index, uint32_t event_count, uint32_t interval_microseconds) {
    if (trigger_client_arr_ == nullptr) {
        LOG4CXX_WARN(logger_, "sender clients are not yet created and running.");
        return false;
    }
    unique_lock<mutex> wait_lk(wait_mtx_);
    const uint32_t last_event_index = start_event_index + event_count - 1;
    for (uint32_t event_index = start_event_index; event_index <= last_event_index; event_index++) {
        duration<double, micro> start_time = system_clock::now().time_since_epoch();
        if (!trigger_one_event(event_index)) {
            LOG4CXX_WARN(logger_, "stop when encountered failed trigger.");
            return false;
        }
        if (last_event_index == event_index) break;
        duration<double, micro> finish_time = system_clock::now().time_since_epoch();
        uint32_t time_used = finish_time.count() - start_time.count();
        if (time_used < interval_microseconds) {
            wait_cv_.wait_for(wait_lk, microseconds(interval_microseconds - time_used));
        }
    }

    return true;
}

bool diffraflow::TrgCoordinator::read_address_list_(const char* filename, vector<pair<string, int>>& addr_vec) {
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