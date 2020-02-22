#include "DynamicConfiguration.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <iostream>
#include <cstdlib>
#include <zookeeper/zookeeper.h>

using std::cout;
using std::endl;
using std::lock_guard;

log4cxx::LoggerPtr diffraflow::DynamicConfiguration::logger_
    = log4cxx::Logger::getLogger("DynamicConfiguration");

diffraflow::DynamicConfiguration::DynamicConfiguration() {

}

diffraflow::DynamicConfiguration::~DynamicConfiguration() {

}

bool diffraflow::DynamicConfiguration::load(const char* filename) {
    vector< pair<string, string> > conf_KV_vec;
    if (!read_conf_KV_vec_(filename, conf_KV_vec)) {
        LOG4CXX_ERROR(logger_, "Failed to read configuration file: " << filename);
        return false;
    }
    lock_guard<mutex> lk(conf_map_mtx_);
    conf_map_.clear();
    for (size_t i = 0; i < conf_KV_vec.size(); i++) {
        string key = conf_KV_vec[i].first;
        string value = conf_KV_vec[i].second;
        conf_map_[key] = value;
    }
    return true;
}

void diffraflow::DynamicConfiguration::print() {
    lock_guard<mutex> lk(conf_map_mtx_);
    for (map<string, string>::iterator iter = conf_map_.begin(); iter != conf_map_.end(); ++iter) {
        cout << "- " << iter->first << " = " << iter->second << endl;
    }
}

void diffraflow::DynamicConfiguration::convert() {

}

/////////////////////////////////////////////////////////////
// for ZooKeeper
/////////////////////////////////////////////////////////////
diffraflow::DynamicConfiguration* diffraflow::DynamicConfiguration::the_obj_ = nullptr;
atomic_bool diffraflow::DynamicConfiguration::zookeeper_intialized_(false);
atomic_int diffraflow::DynamicConfiguration::count_down_(0);
zhandle_t* diffraflow::DynamicConfiguration::zookeeper_handle_(nullptr);
string diffraflow::DynamicConfiguration::zookeeper_server_;
string diffraflow::DynamicConfiguration::zookeeper_root_node_;
string diffraflow::DynamicConfiguration::zookeeper_log_level_;
int diffraflow::DynamicConfiguration::zookeeper_expiration_time_(0);
string diffraflow::DynamicConfiguration::zookeeper_auth_string_;

bool diffraflow::DynamicConfiguration::zookeeper_config(diffraflow::DynamicConfiguration* obj) {
    if (zookeeper_intialized_) {
        LOG4CXX_WARN(logger_, "Close the existing zookeeper session.");
        // close existing zookeeper session
        zookeeper_close(zookeeper_handle_);
        zookeeper_intialized_ = false;
        the_obj_ = nullptr;
    }
    // config for a new zookeeper session
    zookeeper_server_.clear();
    zookeeper_root_node_.clear();
    zookeeper_log_level_ = "info";
    zookeeper_auth_string_.clear();
    zookeeper_expiration_time_ = 10000;  // 10 seconds
    lock_guard<mutex> lk(obj->conf_map_mtx_);
    for (map<string, string>::iterator iter = obj->conf_map_.begin(); iter != obj->conf_map_.end(); ++iter) {
        if (iter->first == "zookeeper_server") {
            zookeeper_server_ = iter->second;
            iter = obj->conf_map_.erase(iter);
        } else if (iter->first == "zookeeper_expiration_time") {
            zookeeper_expiration_time_ = atoi(iter->second.c_str());
            iter = obj->conf_map_.erase(iter);
        } else if (iter->first == "zookeeper_auth_string") {
            zookeeper_auth_string_ = iter->second;
            iter = obj->conf_map_.erase(iter);
        } else if (iter->first == "zookeeper_root_node") {
            zookeeper_root_node_ = iter->second;
            iter = obj->conf_map_.erase(iter);
        } else if (iter->first == "zookeeper_log_level") {
            zookeeper_log_level_ = iter->second;
            iter = obj->conf_map_.erase(iter);
        }
    }
    // check
    bool succ_flag = true;
    if (zookeeper_server_.empty()) {
        LOG4CXX_ERROR(logger_, "zookeeper_server is not set.");
        succ_flag = false;
    }
    if (zookeeper_expiration_time_ < 5000 || zookeeper_expiration_time_ > 15000) {
        zookeeper_expiration_time_ = 10000;
        LOG4CXX_WARN(logger_, "zookeeper_expiration_time is out of range, the default value 10000 is used.");
    }
    // set current object which zookeeper will update
    if (succ_flag) {
        the_obj_ = obj;
    }
    // set zookeeper log level
    if (zookeeper_log_level_ == "debug") {
        zoo_set_debug_level(ZOO_LOG_LEVEL_DEBUG);
    } else if (zookeeper_log_level_ == "info") {
        zoo_set_debug_level(ZOO_LOG_LEVEL_INFO);
    } else if (zookeeper_log_level_ == "warn") {
        zoo_set_debug_level(ZOO_LOG_LEVEL_WARN);
    } else if (zookeeper_log_level_ == "error") {
        zoo_set_debug_level(ZOO_LOG_LEVEL_ERROR);
    }
    // return
    return succ_flag;
}

void diffraflow::DynamicConfiguration::zookeeper_start(bool is_upd) {

}

bool diffraflow::DynamicConfiguration::zookeeper_bootstrap(string parent_node,
    const map<string, string>& cfg_map) {

    return true;
}

bool diffraflow::DynamicConfiguration::zookeeper_update_remote(string parent_node,
    const map<string, string>& cfg_map) {

    return true;
}
