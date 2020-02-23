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
    zookeeper_initialized_ = false;
    zookeeper_handle_ = nullptr;
    zookeeper_expiration_time_ = 10000;
    zookeeper_is_updater_ = false;
    zookeeper_log_level_ = "info";
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
        if (key == "zookeeper_server") {
            zookeeper_server_ = value;
        } else if (key == "zookeeper_expiration_time") {
            zookeeper_expiration_time_ = atoi(value.c_str());
        } else if (key == "zookeeper_auth_string") {
            zookeeper_auth_string_ = value;
        } else if (key == "zookeeper_root_node") {
            zookeeper_root_node_ = value;
        } else if (key == "zookeeper_log_level") {
            zookeeper_log_level_ = value;
        } else {
            conf_map_[key] = value;
        }
    }
    return true;
}

void diffraflow::DynamicConfiguration::print() {
    lock_guard<mutex> lk(conf_map_mtx_);
    for (map<string, string>::iterator iter = conf_map_.begin(); iter != conf_map_.end(); ++iter) {
        cout << "- " << iter->first << " = " << iter->second << endl;
    }
}

void diffraflow::DynamicConfiguration::convert_and_check() {

}

void diffraflow::DynamicConfiguration::zookeeper_start(bool is_upd) {
    if (zookeeper_initialized_) {
        LOG4CXX_WARN(logger_, "Close the existing zookeeper session.");
        // close existing zookeeper session
        zookeeper_close(zookeeper_handle_);
        zookeeper_initialized_ = false;
    }
    // check
    if (zookeeper_server_.empty()) {
        LOG4CXX_ERROR(logger_, "zookeeper_server is not set, stop starting zookeeper session.");
        return;
    }
    if (zookeeper_is_updater_ && zookeeper_auth_string_.empty()) {
        LOG4CXX_ERROR(logger_, "zookeeper_auth_string is not set for updater, stop starting zookeeper session.");
        return;
    }
    if (zookeeper_expiration_time_ < 5000 || zookeeper_expiration_time_ > 15000) {
        zookeeper_expiration_time_ = 10000;
        LOG4CXX_WARN(logger_, "zookeeper_expiration_time is out of range, use the default value 10000.");
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
    } else {
        zoo_set_debug_level(ZOO_LOG_LEVEL_INFO);
        LOG4CXX_WARN(logger_, "an unknown zookeeper_log_level is set: " << zookeeper_log_level_ << ", use info instread.");
    }
    string zk_conn_string = (zookeeper_root_node_.empty() ?
        zookeeper_server_ : zookeeper_server_ + "/" + zookeeper_root_node_);
    zookeeper_handle_ = zookeeper_init(zk_conn_string.c_str(),
        zookeeper_main_watcher_, zookeeper_expiration_time_, NULL, this, 0);
}

bool diffraflow::DynamicConfiguration::zookeeper_create_config(string parent_node,
    const map<string, string>& config_map) {

    return true;
}

bool diffraflow::DynamicConfiguration::zookeeper_change_config(string parent_node,
    const map<string, string>& config_map) {

    return true;
}

bool diffraflow::DynamicConfiguration::zookeeper_watch_config(string parent_node,
    diffraflow::DynamicConfiguration* config_obj) {

}

void diffraflow::DynamicConfiguration::zookeeper_main_watcher_(
    zhandle_t* zh, int type, int state, const char* path, void* context) {
    if (type != ZOO_SESSION_EVENT) return;
    DynamicConfiguration* the_obj = (DynamicConfiguration*) zoo_get_context(zh);
    if (state == ZOO_CONNECTED_STATE) {
        the_obj->zookeeper_initialized_ = true;
        the_obj->zookeeper_initialized_cv_.notify_all();
    } else if (state == ZOO_EXPIRED_SESSION_STATE) {
        LOG4CXX_WARN(logger_, "zookeeper session is expired, try to recreate a session.");
        zookeeper_close(the_obj->zookeeper_handle_);
        the_obj->zookeeper_initialized_ = false;
        the_obj->zookeeper_start(the_obj->zookeeper_is_updater_);
    }
}