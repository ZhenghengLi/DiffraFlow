#include "DynamicConfiguration.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <zookeeper/zookeeper.h>
#include <msgpack.hpp>
#include <ctime>
#include <regex>

using std::cout;
using std::endl;
using std::lock_guard;
using std::unique_lock;
using std::regex;
using std::regex_match;

log4cxx::LoggerPtr diffraflow::DynamicConfiguration::logger_
    = log4cxx::Logger::getLogger("DynamicConfiguration");

diffraflow::DynamicConfiguration::DynamicConfiguration() {
    zookeeper_handle_ = nullptr;
    zookeeper_connected_ = false;
    zookeeper_auth_res_ = kUnknown;
    zookeeper_expiration_time_ = 10000;
    zookeeper_is_updater_ = false;
    zookeeper_log_level_ = "info";
    zookeeper_znode_buffer_cap_ = 1024 * 1024;
    zookeeper_znode_buffer_ = new char[zookeeper_znode_buffer_cap_];
    zookeeper_znode_buffer_len_ = zookeeper_znode_buffer_cap_;
    zookeeper_config_path_ = "/myconfig";
}

diffraflow::DynamicConfiguration::~DynamicConfiguration() {
    if (zookeeper_handle_ != nullptr) {
        zookeeper_stop();
    }
    delete [] zookeeper_znode_buffer_;
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
        } else if (key == "zookeeper_chroot") {
            zookeeper_chroot_ = value;
        } else if (key == "zookeeper_expiration_time") {
            zookeeper_expiration_time_ = atoi(value.c_str());
        } else if (key == "zookeeper_auth_string") {
            zookeeper_auth_string_ = value;
        } else if (key == "zookeeper_log_level") {
            zookeeper_log_level_ = value;
        } else if (key == "zookeeper_config_path") {
            zookeeper_config_path_ = value;
        } else {
            conf_map_[key] = value;
        }
    }
    // check
    if (zookeeper_server_.empty()) {
        LOG4CXX_ERROR(logger_, "zookeeper_server is not set.");
        return false;
    }
    if (!zookeeper_chroot_.empty()) {
        if (!zookeeper_check_path_(zookeeper_chroot_.c_str())) {
            LOG4CXX_ERROR(logger_, "zookeeper_chroot has invalid path, it must start with / and not end with /.")
            return false;
        }
    }
    if (!zookeeper_config_path_.empty()) {
        if (!zookeeper_check_path_(zookeeper_config_path_.c_str())) {
            LOG4CXX_ERROR(logger_, "zookeeper_config_path has invalid path, it must start with / and not end with /.")
            return false;
        }
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

void diffraflow::DynamicConfiguration::zookeeper_print_setting() {
    cout << "zookeeper setting:" << endl;
    cout << "- zookeeper_server = " << zookeeper_server_ << endl;
    cout << "- zookeeper_chroot = " << zookeeper_chroot_ << endl;
    cout << "- zookeeper_expiration_time = " << zookeeper_expiration_time_ << endl;
    cout << "- zookeeper_auth_string = " << zookeeper_auth_string_ << endl;
    cout << "- zookeeper_log_level = " << zookeeper_log_level_ << endl;
    cout << "- zookeeper_config_path = " << zookeeper_config_path_ << endl;
}

void diffraflow::DynamicConfiguration::convert_and_check() {
    LOG4CXX_WARN(logger_, "convert_and_check() is not implemented.")
}

// zookeeper operations =====================

bool diffraflow::DynamicConfiguration::zookeeper_start(bool is_upd) {
    zookeeper_is_updater_ = is_upd;
    if (zookeeper_is_updater_ && zookeeper_auth_string_.empty()) {
        LOG4CXX_ERROR(logger_, "zookeeper_auth_string is not set for updater, zookeeper session does not start.");
        return false;
    }
    return zookeeper_start();
}

bool diffraflow::DynamicConfiguration::zookeeper_start() {
    if (zookeeper_connected_) {
        LOG4CXX_WARN(logger_, "Close the existing zookeeper session.");
        // close existing zookeeper session
        zookeeper_stop();
    }
    string zk_conn_string = (zookeeper_chroot_.empty() ?
        zookeeper_server_ : zookeeper_server_ + zookeeper_chroot_);
    // init zookeeper session
    zookeeper_handle_ = zookeeper_init(zk_conn_string.c_str(),
        zookeeper_main_watcher_, zookeeper_expiration_time_, NULL, this, 0);

    if (zookeeper_is_updater_) {
        zookeeper_add_auth_();
        return zookeeper_authadding_wait_();
    } else {
        zookeeper_connection_wait_();
        return true;
    }
}

void diffraflow::DynamicConfiguration::zookeeper_stop() {
    if (zookeeper_handle_ != nullptr) {
        zookeeper_close(zookeeper_handle_);
        zookeeper_handle_ = nullptr;
    }
    lock_guard<mutex> lk1(zookeeper_connected_mtx_);
    zookeeper_connected_ = false;
    lock_guard<mutex> lk2(zookeeper_auth_res_mtx_);
    zookeeper_auth_res_ = kUnknown;
}

bool diffraflow::DynamicConfiguration::zookeeper_create_config(
    const char* config_path, const map<string, string>& config_map) {
    if (!zookeeper_check_path_(config_path)) {
        LOG4CXX_ERROR(logger_, "config path " << config_path << " is invalid, it must start with / and not end with /.")
        return false;
    }
    // wait for re-reconnecting
    zookeeper_connection_wait_();
    // wait for adding auth
    if (!zookeeper_authadding_wait_()) return false;

    // create config from here.
    static char zkacl_world[] = "world", zkacl_anyone[] = "anyone";
    static char zkacl_auth[] = "auth", zkacl_empty[] = "";
    ACL my_acl[] = {
        {ZOO_PERM_READ, {zkacl_world, zkacl_anyone}},
        {ZOO_PERM_ALL, {zkacl_auth, zkacl_empty}}
    };
    ACL_vector zkacl_vec = {2, my_acl};
    // serialize config_map
    msgpack::sbuffer config_sbuf;
    msgpack::pack(config_sbuf, config_map);
    int rc = zoo_create(zookeeper_handle_,
        config_path, config_sbuf.data(), config_sbuf.size(),
        &zkacl_vec, 0, nullptr, 0);
    switch (rc) {
    case ZOK:
        LOG4CXX_INFO(logger_, "Successfully created config path " << config_path << ".");
        return true;
    case ZNODEEXISTS:
        LOG4CXX_WARN(logger_, "the config path " << config_path << " already exists.");
        return false;
    case ZNONODE:
        LOG4CXX_WARN(logger_, "the parent node of config path " << config_path << " does not exist.");
        return false;
    case ZNOAUTH:
        LOG4CXX_WARN(logger_, "the client does not have permission to create config path " << config_path << ".");
        return false;
    default:
        LOG4CXX_WARN(logger_, "failed to create config path " << config_path << " with error code: " << rc << ".");
        return false;
    }
}

bool diffraflow::DynamicConfiguration::zookeeper_delete_config(const char* config_path) {
    if (!zookeeper_check_path_(config_path)) {
        LOG4CXX_ERROR(logger_, "config path " << config_path << " is invalid, it must start with / and not end with /.")
        return false;
    }
    // wait for re-reconnecting
    zookeeper_connection_wait_();
    // wait for adding auth
    if (!zookeeper_authadding_wait_()) return false;

    // delete config from here
    int rc = zoo_delete(zookeeper_handle_, config_path, -1);
    switch(rc) {
    case ZOK:
        LOG4CXX_INFO(logger_, "Sucessfully deleted config path " << config_path << ".");
        return true;
    case ZNONODE:
        LOG4CXX_WARN(logger_, "the config path " << config_path << " does not exist.");
        return false;
    case ZNOAUTH:
        LOG4CXX_WARN(logger_, "the client does not have permissioin to delete config path " << config_path << ".");
        return false;
    case ZNOTEMPTY:
        LOG4CXX_WARN(logger_, "the config path " << config_path << " has children, it cannot be deleted.");
        return false;
    default:
        LOG4CXX_WARN(logger_, "failed to delete config path " << config_path << " with error code: " << rc << ".");
        return false;
    }
}

bool diffraflow::DynamicConfiguration::zookeeper_change_config(
    const char* config_path, const map<string, string>& config_map) {
    if (!zookeeper_check_path_(config_path)) {
        LOG4CXX_ERROR(logger_, "config path " << config_path << " is invalid, it must start with / and not end with /.")
        return false;
    }
    // wait for re-reconnecting
    zookeeper_connection_wait_();
    // wait for adding auth
    if (!zookeeper_authadding_wait_()) return false;

    // change config from here.
    msgpack::sbuffer config_sbuf;
    msgpack::pack(config_sbuf, config_map);
    int rc = zoo_set(zookeeper_handle_,
        config_path, config_sbuf.data(), config_sbuf.size(), -1);
    switch (rc) {
    case ZOK:
        LOG4CXX_INFO(logger_, "Sucessfully changed the data of config path " << config_path << ".");
        return true;
    case ZNONODE:
        LOG4CXX_WARN(logger_, "the config path " << config_path << " does not exist.");
        return false;
    case ZNOAUTH:
        LOG4CXX_WARN(logger_, "the client does not have permissioin to change the data of config path " << config_path << ".");
        return false;
    default:
        LOG4CXX_WARN(logger_, "failed to change the data of config path " << config_path << " with error code: " << rc << ".");
        return false;
    }
}

bool diffraflow::DynamicConfiguration::zookeeper_fetch_config(
    const char* config_path, map<string, string>& config_map, time_t& config_mtime) {
    if (!zookeeper_check_path_(config_path)) {
        LOG4CXX_ERROR(logger_, "config path " << config_path << " is invalid, it must start with / and not end with /.")
        return false;
    }
    // wait for re-reconnecting
    zookeeper_connection_wait_();

    // fetch config from here.
    lock_guard<mutex> lk(zookeeper_znode_mtx_);
    zookeeper_znode_buffer_len_ = zookeeper_znode_buffer_cap_;
    int rc = zoo_get(zookeeper_handle_, config_path, 0,
        zookeeper_znode_buffer_, &zookeeper_znode_buffer_len_, &zookeeper_znode_stat_);
    switch (rc) {
    case ZOK:
        if (zookeeper_znode_buffer_len_ <= 0) {
            LOG4CXX_WARN(logger_, "There is no data in config path " << config_path << ".");
            return false;
        } else {
            LOG4CXX_INFO(logger_, "Sucessfully fetched the data of config path " << config_path << ".");
            try {
                msgpack::unpack(zookeeper_znode_buffer_,
                    zookeeper_znode_buffer_len_).get().convert(config_map);
            } catch (std::exception& e) {
                LOG4CXX_ERROR(logger_, "Failed to deserialize the data of config_path "
                    << zookeeper_config_path_ << " with exception" << e.what());
                return false;
            }
            config_mtime = zookeeper_znode_stat_.mtime / 1000;
            return true;
        }
    case ZNONODE:
        LOG4CXX_WARN(logger_, "the config path " << config_path << " does not exist.");
        return false;
    case ZNOAUTH:
        LOG4CXX_WARN(logger_, "the client does not have permissioin to fetch the data of config path " << config_path << ".");
        return false;
    default:
        LOG4CXX_WARN(logger_, "failed to fetch the data of config path " << config_path << " with error code: " << rc << ".");
        return false;
    }
}

void diffraflow::DynamicConfiguration::zookeeper_sync_config() {
    // wait for re-connected
    zookeeper_connection_wait_();
    // check existence and watch
    zoo_awexists(zookeeper_handle_, zookeeper_config_path_.c_str(),
        zookeeper_config_watcher_, this,
        zookeeper_stat_completion_, this);
}

void diffraflow::DynamicConfiguration::zookeeper_add_auth_() {
    // wait for connected
    zookeeper_connection_wait_();
    // add auth
    zoo_add_auth(zookeeper_handle_, "digest",
        zookeeper_auth_string_.c_str(), zookeeper_auth_string_.length(),
        zookeeper_auth_completion_, this);
}

void diffraflow::DynamicConfiguration::zookeeper_get_config_() {
    // wait for re-connected
    zookeeper_connection_wait_();
    // get znode data
    zoo_aget(zookeeper_handle_, zookeeper_config_path_.c_str(), 0,
        zookeeper_data_completion_, this);
}

void diffraflow::DynamicConfiguration::zookeeper_connection_wait_() {
    unique_lock<mutex> lk(zookeeper_connected_mtx_);
    bool log_flag = (!zookeeper_connected_);
    zookeeper_connected_cv_.wait(lk,
        [this]() {return zookeeper_connected_;});
    if (log_flag) {
        LOG4CXX_INFO(logger_, "zookeeper session is set up.");
    }
}

bool diffraflow::DynamicConfiguration::zookeeper_authadding_wait_() {
    if (!zookeeper_is_updater_) {
        LOG4CXX_ERROR(logger_, "Current object is not an updater");
        return false;
    }
    unique_lock<mutex> lk(zookeeper_auth_res_mtx_);
    bool log_flag = (zookeeper_auth_res_ == kUnknown);
    zookeeper_auth_res_cv_.wait(lk,
        [this]() {return zookeeper_auth_res_ != kUnknown;});
    if (zookeeper_auth_res_ == kSucc) {
        if (log_flag) {
            LOG4CXX_INFO(logger_, "Successfully add digest auth: " << zookeeper_auth_string_);
        }
        return true;
    } else {
        if (log_flag) {
            LOG4CXX_ERROR(logger_, "Failed to add digest auth: " << zookeeper_auth_string_);
        }
        return false;
    }
}

bool diffraflow::DynamicConfiguration::zookeeper_check_path_(const char* path) {
    regex start_with_slash("^\\s*/.*");
    regex end_with_slash(".*/\\s*$");
    if (regex_match(path, start_with_slash) && !regex_match(path, end_with_slash)) {
        return true;
    } else {
        return false;
    }
}

// zookeeper callbacks =====================

void diffraflow::DynamicConfiguration::zookeeper_main_watcher_(
    zhandle_t* zh, int type, int state, const char* path, void* context) {
    if (type != ZOO_SESSION_EVENT) return;
    DynamicConfiguration* the_obj = (DynamicConfiguration*) zoo_get_context(zh);
    unique_lock<mutex> lk(the_obj->zookeeper_connected_mtx_);
    if (state == ZOO_CONNECTED_STATE) {
        the_obj->zookeeper_connected_ = true;
        the_obj->zookeeper_connected_cv_.notify_all();
    } else if (state == ZOO_CONNECTING_STATE) {
        the_obj->zookeeper_connected_ = false;
    } else if (state == ZOO_EXPIRED_SESSION_STATE) {
        LOG4CXX_WARN(logger_, "zookeeper session is expired, try to recreate a session.");
        lk.unlock();
        the_obj->zookeeper_stop();
        the_obj->zookeeper_start();
    } else {
        LOG4CXX_WARN(logger_, "zookeeper session state with error code: " << state);
    }
}

void diffraflow::DynamicConfiguration::zookeeper_auth_completion_(int rc, const void* data) {
    DynamicConfiguration* the_obj = (DynamicConfiguration*) data;
    unique_lock<mutex> lk(the_obj->zookeeper_auth_res_mtx_);
    switch (rc) {
    case ZOK:
        the_obj->zookeeper_auth_res_ = kSucc;
        the_obj->zookeeper_auth_res_cv_.notify_all();
        break;
    case ZCONNECTIONLOSS:
    case ZOPERATIONTIMEOUT:
        lk.unlock();
        the_obj->zookeeper_add_auth_();
        break;
    default:
        LOG4CXX_WARN(logger_, "error found when authing with error code: " << rc);
        the_obj->zookeeper_auth_res_ = kFail;
        the_obj->zookeeper_auth_res_cv_.notify_all();
    }
}

void diffraflow::DynamicConfiguration::zookeeper_config_watcher_(
    zhandle_t* zh, int type, int state, const char* path, void* context) {
    DynamicConfiguration* the_obj = (DynamicConfiguration*) zoo_get_context(zh);
    if (type == ZOO_CHANGED_EVENT || type == ZOO_CREATED_EVENT || type == ZOO_DELETED_EVENT) {
        the_obj->zookeeper_sync_config();
    }
}

void diffraflow::DynamicConfiguration::zookeeper_stat_completion_(int rc,
    const struct Stat* stat, const void* data) {
    DynamicConfiguration* the_obj = (DynamicConfiguration*) data;
    switch (rc) {
    case ZOK:
        the_obj->zookeeper_get_config_();
    case ZCONNECTIONLOSS:
    case ZOPERATIONTIMEOUT:
        the_obj->zookeeper_sync_config();
    case ZNONODE:
        LOG4CXX_WARN(logger_, "the config znode of path " << the_obj->zookeeper_config_path_
            << "does not exist or it is deleted.");
        break;
    default:
        LOG4CXX_WARN(logger_, "error found when checking existence of path "
            << the_obj->zookeeper_config_path_ << " with error code: " << rc);
    }
}

void diffraflow::DynamicConfiguration::zookeeper_data_completion_(int rc, const char *value,
    int value_len, const struct Stat *stat, const void *data) {
    DynamicConfiguration* the_obj = (DynamicConfiguration*) data;
    switch (rc) {
    case ZOK:
        if (value_len <= 0) {
            LOG4CXX_WARN(logger_, "there is no data in znode of path " << the_obj->zookeeper_config_path_ << ".");
        } else {
            lock_guard<mutex> lk(the_obj->conf_map_mtx_);
            // znode data -> conf_map_
            try {
                msgpack::unpack(value, value_len).get().convert(the_obj->conf_map_);
            } catch (const std::exception& e) {
                LOG4CXX_ERROR(logger_, "failed to deserialize the data of config_path "
                    << the_obj->zookeeper_config_path_ << " with exception" << e.what());
                return;
            }
            the_obj->conf_map_mtime_ = stat->mtime / 1000;
            // conf_map_ -> config fields with proper types and units
            the_obj->convert_and_check();
            LOG4CXX_INFO(logger_, "Successfully synchronized config data with mtime: "
                << ctime(&the_obj->conf_map_mtime_));
        }
        break;
    case ZCONNECTIONLOSS:
    case ZOPERATIONTIMEOUT:
        the_obj->zookeeper_get_config_();
        break;
    case ZNONODE:
        LOG4CXX_WARN(logger_, "there is no znode of path: "
            << the_obj->zookeeper_config_path_ << ".");
        break;
    default:
        LOG4CXX_WARN(logger_, "error found when reading path "
            << the_obj->zookeeper_config_path_ << " with error code: " << rc);
    }
}
