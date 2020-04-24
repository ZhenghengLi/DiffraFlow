#include "CtrHttpServer.hh"
#include "CtrMonLoadBalancer.hh"
#include "DynamicConfiguration.hh"

#include <regex>
#include <queue>
#include <map>
#include <algorithm>

using namespace web;
using namespace http;
using namespace experimental::listener;

using std::lock_guard;
using std::unique_lock;
using std::regex;
using std::regex_match;
using std::regex_replace;
using std::queue;
using std::map;
using std::pair;

log4cxx::LoggerPtr diffraflow::CtrHttpServer::logger_
    = log4cxx::Logger::getLogger("CtrHttpServer");

diffraflow::CtrHttpServer::CtrHttpServer(CtrMonLoadBalancer* mon_ld_bl, DynamicConfiguration* zk_conf_client) {
    listener_ = nullptr;
    server_status_ = kNotStart;
    monitor_load_balancer_ = mon_ld_bl;
    zookeeper_config_client_ = zk_conf_client;
}

diffraflow::CtrHttpServer::~CtrHttpServer() {

}

bool diffraflow::CtrHttpServer::start(string host, int port) {
    if (server_status_ == kRunning) {
        LOG4CXX_WARN(logger_, "http server has already been started.");
        return false;
    }
    uri_builder uri_b;
    uri_b.set_scheme("http");
    uri_b.set_host(host);
    uri_b.set_port(port);
    listener_ = new http_listener(uri_b.to_uri());
    listener_->support(methods::GET, std::bind(&CtrHttpServer::handleGet_, this, std::placeholders::_1));
    listener_->support(methods::POST, std::bind(&CtrHttpServer::handlePost_, this, std::placeholders::_1));
    listener_->support(methods::PUT, std::bind(&CtrHttpServer::handlePut_, this, std::placeholders::_1));
    listener_->support(methods::PATCH, std::bind(&CtrHttpServer::handlePatch_, this, std::placeholders::_1));
    listener_->support(methods::DEL, std::bind(&CtrHttpServer::handleDel_, this, std::placeholders::_1));

    try {
        listener_->open().wait();
        server_status_ = kRunning;
        return true;
    } catch (std::exception& e) {
        LOG4CXX_ERROR(logger_, "failed to start http server: " << e.what());
        return false;
    } catch(...) {
        LOG4CXX_ERROR(logger_, "failed to start http server with unknown error.");
        return false;
    }
    return true;
}

void diffraflow::CtrHttpServer::stop() {
    if (listener_ == nullptr) return;

    try {
        listener_->close().wait();
    } catch (std::exception& e) {
        LOG4CXX_WARN(logger_, "exception found when closing http listener: " << e.what());
    } catch (...) {
        LOG4CXX_WARN(logger_, "an unknown exception found when closing http listener.");
    }

    delete listener_;
    listener_ = nullptr;

    server_status_ = kStopped;
    cv_status_.notify_all();

}

void diffraflow::CtrHttpServer::wait() {
    unique_lock<mutex> ulk(mtx_status_);
    cv_status_.wait(ulk,
        [this]() {return server_status_ != kRunning;}
    );
}

void diffraflow::CtrHttpServer::handleGet_(http_request message) {

    vector<utility::string_t> path_vec = uri::split_path(message.relative_uri().path());

    std::regex number_regex("\\d+");

    http_response response;
    if (path_vec.empty()) {
        json::value paths_json;
        paths_json[0] = json::value::string("/event");
        paths_json[1] = json::value::string("/event/<unsigned long>");
        paths_json[2] = json::value::string("/config");
        json::value root_json;
        root_json["paths"] = paths_json;
        message.reply(status_codes::OK, root_json);
        return;
    } else if (path_vec.size() > 2) {
        message.reply(status_codes::NotFound);
        return;
    }

    string request_type = path_vec[0];
    string request_value = (path_vec.size() > 1 ? path_vec[1] : "");

    if (request_type == "event") {
        if (!request_value.empty()) {
            if (!regex_match(request_value, number_regex)) {
                message.reply(status_codes::NotFound);
                return;
            }
        }
        if (monitor_load_balancer_->do_one_request(response, request_value)) {
            message.reply(response);
        } else {
            message.reply(status_codes::NotFound);
        }
    } else if (request_type == "config") {
        if (request_value.empty()) {
            vector<string> config_list;
            int zoo_err = zookeeper_config_client_->zookeeper_get_children("/", config_list);
            if (zoo_err == ZOK) {
                json::value config_list_json;
                for (size_t i = 0; i < config_list.size(); i++) {
                    config_list_json[i] = json::value::string(config_list[i]);
                }
                json::value root_json;
                root_json["config_list"] = config_list_json;
                message.reply(status_codes::OK, root_json);
            } else {
                message.reply(status_codes::InternalError);
            }
        } else {
            map<string, string> config_map;
            time_t config_mtime;
            int version;
            int zoo_err = zookeeper_config_client_->zookeeper_fetch_config(
                request_value.c_str(), config_map, config_mtime, version);
            if (zoo_err == ZOK) {
                json::value config_map_json;
                for (const pair<string, string>& item: config_map) {
                    config_map_json[item.first] = json::value::string(item.second);
                }
                json::value root_json;
                root_json["name"] = json::value::string(request_value);
                root_json["data"] = config_map_json;
                message.reply(status_codes::OK, root_json);
            } else if (zoo_err == ZNONODE) {
                message.reply(status_codes::NotFound);
            } else {
                message.reply(status_codes::InternalError);
            }
        }
    } else {
        message.reply(status_codes::NotFound);
    }
}

void diffraflow::CtrHttpServer::handlePost_(http_request message) {
    vector<utility::string_t> path_vec = uri::split_path(message.relative_uri().path());
    http_response response;
    if (path_vec.size() != 2) {
        message.reply(status_codes::BadRequest);
        return;
    }
    string request_type = path_vec[0];
    string request_value = path_vec[1];
    if (request_type != "config") {
        message.reply(status_codes::BadRequest);
        return;
    }
    string znode_path = string("/") + request_value;
    json::value request_body_json = message.extract_json().get();
    map<string, string> config_map;
    if (request_body_json.is_object()) {
        json::object json_object = request_body_json.as_object();
        for (json::object::iterator iter = json_object.begin(); iter != json_object.end(); ++iter) {
            if (iter->second.is_string()) {
                config_map[iter->first] = iter->second.as_string();
            } else {
                message.reply(status_codes::BadRequest, utility::string_t("config value should only be of type string."));
                return;
            }
        }
        int zoo_err = zookeeper_config_client_->zookeeper_create_config(znode_path.c_str(), config_map);
        if (zoo_err == ZOK) {
            message.reply(status_codes::OK);
        } else if (zoo_err == ZNODEEXISTS)  {
            message.reply(status_codes::Conflict, utility::string_t("config name already exists."));
        } else {
            message.reply(status_codes::InternalError);
        }
    } else {
        message.reply(status_codes::BadRequest, utility::string_t("request body should be a json object."));
    }
}

void diffraflow::CtrHttpServer::handlePut_(http_request message) {
    vector<utility::string_t> path_vec = uri::split_path(message.relative_uri().path());
    http_response response;
    if (path_vec.size() != 2) {
        message.reply(status_codes::BadRequest);
        return;
    }
    string request_type = path_vec[0];
    string request_value = path_vec[1];
    if (request_type != "config") {
        message.reply(status_codes::BadRequest);
        return;
    }
    string znode_path = string("/") + request_value;
    json::value request_body_json = message.extract_json().get();
    map<string, string> config_map;
    if (request_body_json.is_object()) {
        json::object json_object = request_body_json.as_object();
        for (json::object::iterator iter = json_object.begin(); iter != json_object.end(); ++iter) {
            if (iter->second.is_string()) {
                config_map[iter->first] = iter->second.as_string();
            } else {
                message.reply(status_codes::BadRequest, utility::string_t("config value should only be of type string."));
                return;
            }
        }
        int zoo_err = zookeeper_config_client_->zookeeper_change_config(znode_path.c_str(), config_map);
        if (zoo_err == ZOK) {
            message.reply(status_codes::OK);
        } else if (zoo_err == ZNONODE)  {
            message.reply(status_codes::BadRequest, utility::string_t("config name does not exist."));
        } else {
            message.reply(status_codes::InternalError);
        }
    } else {
        message.reply(status_codes::BadRequest, utility::string_t("request body should be a json object."));
    }
}

void diffraflow::CtrHttpServer::handlePatch_(http_request message) {
    vector<utility::string_t> path_vec = uri::split_path(message.relative_uri().path());
    http_response response;
    if (path_vec.size() != 2) {
        message.reply(status_codes::BadRequest);
        return;
    }
    string request_type = path_vec[0];
    string request_value = path_vec[1];
    if (request_type != "config") {
        message.reply(status_codes::BadRequest);
        return;
    }
    string znode_path = string("/") + request_value;
    json::value request_body_json = message.extract_json().get();
    map<string, string> config_map_patch;
    if (request_body_json.is_object()) {
        json::object json_object = request_body_json.as_object();
        for (json::object::iterator iter = json_object.begin(); iter != json_object.end(); ++iter) {
            if (iter->second.is_string()) {
                config_map_patch[iter->first] = iter->second.as_string();
            } else {
                message.reply(status_codes::BadRequest, utility::string_t("config value should only be of type string."));
                return;
            }
        }
        map<string, string> config_map;
        int version = -1;
        time_t config_mtime;
        // fetch
        int zoo_err = zookeeper_config_client_->zookeeper_fetch_config(znode_path.c_str(), config_map, config_mtime, version);
        if (zoo_err == ZNONODE) {
            message.reply(status_codes::BadRequest, utility::string_t("config name does not exist."));
            return;
        } else if (zoo_err != ZOK) {
            message.reply(status_codes::InternalError);
            return;
        }
        // patch
        for (const pair<string, string>& item: config_map_patch) {
            config_map[item.first] = item.second;
        }
        // update
        zoo_err = zookeeper_config_client_->zookeeper_change_config(znode_path.c_str(), config_map, version);
        if (zoo_err == ZOK) {
            message.reply(status_codes::OK);
        } else if (zoo_err == ZNONODE)  {
            message.reply(status_codes::InternalError, utility::string_t("config name may be deleted during patching."));
        } else if (zoo_err == ZBADVERSION)  {
            message.reply(status_codes::InternalError, utility::string_t("config name may be updated during patching."));
        } else {
            message.reply(status_codes::InternalError);
        }
    } else {
        message.reply(status_codes::BadRequest, utility::string_t("request body should be a json object."));
    }
}

void diffraflow::CtrHttpServer::handleDel_(http_request message) {
    vector<utility::string_t> path_vec = uri::split_path(message.relative_uri().path());
    http_response response;
    if (path_vec.size() != 2) {
        message.reply(status_codes::BadRequest);
        return;
    }
    string request_type = path_vec[0];
    string request_value = path_vec[1];
    if (request_type != "config") {
        message.reply(status_codes::BadRequest);
        return;
    }
    string znode_path = string("/") + request_value;
    int zoo_err = zookeeper_config_client_->zookeeper_delete_config(znode_path.c_str());
    if (zoo_err == ZOK) {
        message.reply(status_codes::OK);
    } else if (zoo_err == ZNONODE) {
        message.reply(status_codes::BadRequest, utility::string_t("config name does not exist."));
    } else {
        message.reply(status_codes::InternalError);
    }
}
