#include "DynamicConfiguration.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <zookeeper/zookeeper.h>

using std::cout;
using std::endl;
using std::lock_guard;
using std::unique_lock;

log4cxx::LoggerPtr diffraflow::DynamicConfiguration::logger_
    = log4cxx::Logger::getLogger("DynamicConfiguration");

diffraflow::DynamicConfiguration::DynamicConfiguration() {
    zookeeper_handle_ = nullptr;
    zookeeper_connected_ = false;
    zookeeper_auth_res_ = kUnknown;
    zookeeper_expiration_time_ = 10000;
    zookeeper_is_updater_ = false;
    zookeeper_log_level_ = "info";
    zookeeper_config_path_ = "/myconfig";
    zookeeper_data_mtime_ = 0;
}

diffraflow::DynamicConfiguration::~DynamicConfiguration() {
    if (zookeeper_handle_ != nullptr) {
        zookeeper_stop();
    }
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
        } else if (key == "zookeeper_root_path") {
            zookeeper_root_path_ = value;
        } else if (key == "zookeeper_log_level") {
            zookeeper_log_level_ = value;
        } else {
            conf_map_[key] = value;
        }
    }
    // correction
    if (zookeeper_expiration_time_ < 5000 || zookeeper_expiration_time_ > 15000) {
        zookeeper_expiration_time_ = 10000;
        LOG4CXX_WARN(logger_, "zookeeper_expiration_time is out of range, use the default value 10000.");
    }
    // return
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

// zookeeper operations =====================

void diffraflow::DynamicConfiguration::zookeeper_start(bool is_upd) {
    if (zookeeper_connected_) {
        LOG4CXX_WARN(logger_, "Close the existing zookeeper session.");
        // close existing zookeeper session
        zookeeper_stop();
    }
    zookeeper_is_updater_ = is_upd;
    // check
    if (zookeeper_server_.empty()) {
        LOG4CXX_ERROR(logger_, "zookeeper_server is not set, stop starting zookeeper session.");
        return;
    }
    if (zookeeper_is_updater_ && zookeeper_auth_string_.empty()) {
        LOG4CXX_ERROR(logger_, "zookeeper_auth_string is not set for updater, stop starting zookeeper session.");
        return;
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
    string zk_conn_string = (zookeeper_root_path_.empty() ?
        zookeeper_server_ : zookeeper_server_ + "/" + zookeeper_root_path_);
    // init zookeeper session
    zookeeper_handle_ = zookeeper_init(zk_conn_string.c_str(),
        zookeeper_main_watcher_, zookeeper_expiration_time_, NULL, this, 0);
    // wait for session ready
    {
        unique_lock<mutex> lk(zookeeper_connected_mtx_);
        zookeeper_connected_cv_.wait(lk,
            [this]() {return zookeeper_connected_;});
        LOG4CXX_INFO(logger_, "zookeeper session is set up.");
    }
    if (zookeeper_is_updater_) {
        // add auth for updater
        zoo_add_auth(zookeeper_handle_, "digest",
            zookeeper_auth_string_.c_str(), zookeeper_auth_string_.length(),
            zookeeper_auth_completion_, this);
        // wait for auth completion
        {
            unique_lock<mutex> lk(zookeeper_auth_res_mtx_);
            zookeeper_auth_res_cv_.wait(lk,
                [this]() {return zookeeper_auth_res_ != kUnknown;});
            if (zookeeper_auth_res_ == kSucc) {
                LOG4CXX_INFO(logger_, "Successfully add digest auth: " << zookeeper_auth_string_);
            } else {
                LOG4CXX_ERROR(logger_, "Failed to add digest auth: " << zookeeper_auth_string_);
            }
        }
    }
}

void diffraflow::DynamicConfiguration::zookeeper_stop() {
    if (zookeeper_handle_ != nullptr) {
        zookeeper_close(zookeeper_handle_);
        zookeeper_handle_ = nullptr;
    }
    zookeeper_connected_ = false;
    zookeeper_auth_res_ = kUnknown;
}

bool diffraflow::DynamicConfiguration::zookeeper_create_config(
    const char* config_path, const map<string, string>& config_map) {
    if (!zookeeper_is_updater_) {
        LOG4CXX_ERROR(logger_, "current object is not an updater");
        return false;
    }
    // wait for authorized
    {
        unique_lock<mutex> lk(zookeeper_auth_res_mtx_);
        zookeeper_auth_res_cv_.wait(lk,
            [this]() {return zookeeper_auth_res_ != kUnknown;});
        if (zookeeper_auth_res_ == kFail) {
            LOG4CXX_ERROR(logger_, "zookeeper session is not authorized for updater.");
            return false;
        }
    }
    // create config from here.

    return true;
}

bool diffraflow::DynamicConfiguration::zookeeper_change_config(
    const char* config_path, const map<string, string>& config_map) {
    if (!zookeeper_is_updater_) {
        LOG4CXX_ERROR(logger_, "current object is not an updater");
        return false;
    }
    // wait for authorized
    {
        unique_lock<mutex> lk(zookeeper_auth_res_mtx_);
        zookeeper_auth_res_cv_.wait(lk,
            [this]() {return zookeeper_auth_res_ != kUnknown;});
        if (zookeeper_auth_res_ == kFail) {
            LOG4CXX_ERROR(logger_, "zookeeper session is not authorized for updater.");
            return false;
        }
    }
    // change config from here.

    return true;
}

bool diffraflow::DynamicConfiguration::zookeeper_sync_config() {
    // wait for connected
    {
        unique_lock<mutex> lk(zookeeper_connected_mtx_);
        zookeeper_connected_cv_.wait(lk,
            [this]() {return zookeeper_connected_;});
    }
    zookeeper_data_res_ = kUnknown;
    zoo_awget(zookeeper_handle_, zookeeper_config_path_.c_str(),
        zookeeper_config_watcher_, this,
        zookeeper_data_completion_, this);
    // wait for completion
    {
        unique_lock<mutex> lk(zookeeper_data_res_mtx_);
        zookeeper_data_res_cv_.wait(lk,
            [this]() {return zookeeper_data_res_ != kUnknown;});
        if (zookeeper_data_res_ == kFail) {
            LOG4CXX_ERROR(logger_, "failed to read data from path " << zookeeper_config_path_);
            return false;
        }
    }
    // deserialization: zookeeper_data_string_ -> conf_map_

    // call convert_and_check
    convert_and_check();

    return true;
}

// zookeeper callbacks =====================

void diffraflow::DynamicConfiguration::zookeeper_main_watcher_(
    zhandle_t* zh, int type, int state, const char* path, void* context) {
    if (type != ZOO_SESSION_EVENT) return;
    DynamicConfiguration* the_obj = (DynamicConfiguration*) zoo_get_context(zh);
    if (state == ZOO_CONNECTED_STATE) {
        the_obj->zookeeper_connected_ = true;
        the_obj->zookeeper_connected_cv_.notify_all();
    } else if (state == ZOO_CONNECTING_STATE) {
        the_obj->zookeeper_connected_ = false;
    } else if (state == ZOO_EXPIRED_SESSION_STATE) {
        LOG4CXX_WARN(logger_, "zookeeper session is expired, try to recreate a session.");
        the_obj->zookeeper_stop();
        the_obj->zookeeper_start(the_obj->zookeeper_is_updater_);
    } else {
        LOG4CXX_WARN(logger_, "zookeeper session state with error code: " << state);
    }
}

void diffraflow::DynamicConfiguration::zookeeper_config_watcher_(
    zhandle_t* zh, int type, int state, const char* path, void* context) {
    DynamicConfiguration* the_obj = (DynamicConfiguration*) zoo_get_context(zh);
    if (type == ZOO_CHANGED_EVENT) {
        the_obj->zookeeper_sync_config();
    }
}

void diffraflow::DynamicConfiguration::zookeeper_auth_completion_(int rc, const void* data) {
    DynamicConfiguration* the_obj = (DynamicConfiguration*) data;
    switch (rc) {
    case ZOK:
        the_obj->zookeeper_auth_res_ = kSucc;
        the_obj->zookeeper_auth_res_cv_.notify_all();
        break;
    case ZCONNECTIONLOSS:
    case ZOPERATIONTIMEOUT:
        zoo_add_auth(the_obj->zookeeper_handle_, "digest",
            the_obj->zookeeper_auth_string_.c_str(), the_obj->zookeeper_auth_string_.length(),
            zookeeper_auth_completion_, the_obj);
        break;
    default:
        LOG4CXX_WARN(logger_, "error found when authing with error code: " << rc);
        the_obj->zookeeper_auth_res_ = kFail;
        the_obj->zookeeper_auth_res_cv_.notify_all();
    }
}

void diffraflow::DynamicConfiguration::zookeeper_data_completion_(int rc, const char *value,
    int value_len, const struct Stat *stat, const void *data) {
    DynamicConfiguration* the_obj = (DynamicConfiguration*) data;
    switch (rc) {
    case ZOK:
        the_obj->zookeeper_data_string_.assign(value, value_len);
        the_obj->zookeeper_data_mtime_ = stat->mtime;
        the_obj->zookeeper_data_res_ = kSucc;
        the_obj->zookeeper_data_res_cv_.notify_all();
        break;
    case ZCONNECTIONLOSS:
    case ZOPERATIONTIMEOUT:
        the_obj->zookeeper_sync_config();
        break;
    default:
        LOG4CXX_WARN(logger_, "error found when reading path "
            << the_obj->zookeeper_config_path_ << " with error code: " << rc);
        the_obj->zookeeper_data_res_ = kFail;
        the_obj->zookeeper_data_res_cv_.notify_all();
    }
}
