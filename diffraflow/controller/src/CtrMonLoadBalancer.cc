#include "CtrMonLoadBalancer.hh"
#include <boost/algorithm/string.hpp>

using namespace web;
using namespace http;
using std::ifstream;
using std::lock_guard;

log4cxx::LoggerPtr diffraflow::CtrMonLoadBalancer::logger_ = log4cxx::Logger::getLogger("CtrMonLoadBalancer");

diffraflow::CtrMonLoadBalancer::CtrMonLoadBalancer() { current_index_ = 0; }

diffraflow::CtrMonLoadBalancer::~CtrMonLoadBalancer() {}

bool diffraflow::CtrMonLoadBalancer::create_monitor_clients(const char* filename, int timeout) {
    monitor_clients_vec_.clear();
    ifstream addr_file;
    addr_file.open(filename);
    if (!addr_file.is_open()) {
        LOG4CXX_ERROR(logger_, "address file open failed.");
        return false;
    }
    http_client_config client_config;
    client_config.set_timeout(std::chrono::milliseconds(timeout));
    string oneline;
    const char* node_name_cstr = getenv("NODE_NAME");
    const char* node_ip_cstr = getenv("NODE_IP");
    while (true) {
        oneline = "";
        getline(addr_file, oneline);
        if (addr_file.eof()) break;
        // skip comments and empty lines
        boost::trim(oneline);
        if (oneline[0] == '#') continue;
        if (oneline.length() == 0) continue;
        // construct http client
        try {
            uri uri_val(oneline);
            http_client client(uri_val, client_config);
            monitor_clients_vec_.push_back(client);
            LOG4CXX_INFO(logger_, "created monitor client for " << uri_val.to_string());
        } catch (std::exception& e) {
            LOG4CXX_ERROR(logger_, "exception found when creating monitor client for " << oneline << ": " << e.what());
            return false;
        }
    }
    if (monitor_clients_vec_.size() > 0) {
        return true;
    } else {
        LOG4CXX_ERROR(logger_, "no valid monitor addresses found in file: " << filename);
        return false;
    }
    return true;
}

bool diffraflow::CtrMonLoadBalancer::do_one_request(http_response& response, string event_time_string) {
    lock_guard<mutex> lg(mtx_client_);

    if (monitor_clients_vec_.empty()) {
        LOG4CXX_WARN(logger_, "empty ingester clients list.");
        return false;
    }

    for (size_t addr_idx = current_index_; true;) {
        bool found_exception = false;
        try {
            response = monitor_clients_vec_[addr_idx].request(methods::GET, event_time_string).get();
        } catch (std::exception& e) {
            found_exception = true;
            LOG4CXX_WARN(logger_, "exception found when requesting data from \""
                                      << monitor_clients_vec_[addr_idx].base_uri().to_string() << "\": " << e.what());
        }
        addr_idx++;
        if (addr_idx >= monitor_clients_vec_.size()) {
            addr_idx = 0;
        }
        if (!found_exception && response.status_code() == status_codes::OK) { // succ
            current_index_ = addr_idx;
            return true;
        } else if (addr_idx == current_index_) {
            return false;
        }
    }
}
