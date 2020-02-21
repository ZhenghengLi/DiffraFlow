#include "DynamicConfiguration.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <iostream>
#include <cstdlib>

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
string diffraflow::DynamicConfiguration::zookeeper_server_;
int diffraflow::DynamicConfiguration::zookeeper_expiration_time_(0);
string diffraflow::DynamicConfiguration::zookeeper_auth_string_;

bool diffraflow::DynamicConfiguration::config_zookeeper(diffraflow::DynamicConfiguration* obj) {
    if (zookeeper_intialized_) {
        LOG4CXX_WARN(logger_, "Close the existing zookeeper session.");
        // close existing zookeeper session
    }
    // config for a new zookeeper session
    zookeeper_server_.clear();
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
    // set the object which this zookeeper session will update and return
    if (succ_flag) {
        the_obj_ = obj;
        return true;
    } else {
        the_obj_ = nullptr;
        return false;
    }
}