#include "MonConfig.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <sstream>
#include <ctime>
#include <cstdlib>
#include <thread>
#include <regex>

using std::cout;
using std::endl;
using std::lock_guard;
using std::regex;
using std::regex_match;
using std::regex_replace;

log4cxx::LoggerPtr diffraflow::MonConfig::logger_ = log4cxx::Logger::getLogger("MonConfig");

diffraflow::MonConfig::MonConfig() {
    monitor_id = 0;
    image_http_host = "localhost";
    image_http_port = -1;
    request_timeout = 1000;
    zookeeper_setting_ready_flag_ = false;
    // initial values of dynamic configurations
    dy_param_int_ = 20;
    dy_param_double_ = 100;
    dy_param_string_ = "xfel";

    metrics_pulsar_report_period = 1000;
    metrics_http_port = -1;
}

diffraflow::MonConfig::~MonConfig() {}

bool diffraflow::MonConfig::load(const char* filename) {
    list<pair<string, string>> conf_KV_list;
    if (!read_conf_KV_list_(filename, conf_KV_list)) {
        LOG4CXX_ERROR(logger_, "Failed to read configuration file: " << filename);
        return false;
    }
    // parse
    if (zookeeper_parse_setting_(conf_KV_list)) {
        zookeeper_setting_ready_flag_ = true;
    } else {
        LOG4CXX_WARN(logger_, "zookeeper setting is not ready, configuration will not be dynamically updated.")
    }
    map<string, string> dy_conf_map;
    for (list<pair<string, string>>::iterator iter = conf_KV_list.begin(); iter != conf_KV_list.end(); ++iter) {
        string key = iter->first;
        string value = iter->second;
        // for static parameters
        if (key == "monitor_id") {
            monitor_id = atoi(value.c_str());
        } else if (key == "image_http_host") {
            image_http_host = value;
        } else if (key == "image_http_port") {
            image_http_port = atoi(value.c_str());
        } else if (key == "request_timeout") {
            request_timeout = atoi(value.c_str());
        } else if (key == "metrics_pulsar_broker_address") {
            metrics_pulsar_broker_address = value;
        } else if (key == "metrics_pulsar_topic_name") {
            metrics_pulsar_topic_name = value;
        } else if (key == "metrics_pulsar_message_key") {
            metrics_pulsar_message_key = value;
        } else if (key == "metrics_pulsar_report_period") {
            metrics_pulsar_report_period = atoi(value.c_str());
        } else if (key == "metrics_http_host") {
            metrics_http_host = value;
        } else if (key == "metrics_http_port") {
            metrics_http_port = atoi(value.c_str());
            // for dynamic parameters
        } else {
            dy_conf_map[key] = value;
        }
    }
    // set node name
    const char* node_name_cstr = getenv("NODE_NAME");
    if (node_name_cstr != NULL) {
        node_name = boost::to_upper_copy<string>(node_name_cstr);
    } else {
        node_name = "NODENAME";
    }

    if (node_name_cstr != nullptr && regex_match(metrics_pulsar_message_key, regex(".*NODE_NAME.*"))) {
        metrics_pulsar_message_key = regex_replace(metrics_pulsar_message_key, regex("NODE_NAME"), node_name_cstr);
    }

    // correction
    if (metrics_pulsar_report_period < 500) {
        LOG4CXX_WARN(logger_, "pulsar_report_period < 500, use 500 instead.");
        metrics_pulsar_report_period = 500;
    }
    if (request_timeout < 10) {
        LOG4CXX_WARN(logger_, "request_timeout is too small (< 10), use 10 instead.");
        request_timeout = 10;
    }

    // validation check for static parameters
    bool succ_flag = true;
    if (monitor_id < 0) {
        LOG4CXX_ERROR(logger_, "invalid monitor_id: " << monitor_id);
        succ_flag = false;
    }
    if (image_http_port < 0) {
        LOG4CXX_ERROR(logger_, "invalid image_http_port: " << image_http_port);
        succ_flag = false;
    }
    // check and commit for dynamic parameters
    if (!check_and_commit_(dy_conf_map, time(NULL))) {
        LOG4CXX_ERROR(logger_, "dynamic configurations have invalid values.");
        succ_flag = false;
    }

    if (succ_flag) {

        static_config_json_["node_name"] = json::value::string(node_name);
        static_config_json_["monitor_id"] = json::value::number(monitor_id);
        static_config_json_["image_http_host"] = json::value::string(image_http_host);
        static_config_json_["image_http_port"] = json::value::number(image_http_port);
        static_config_json_["request_timeout"] = json::value::number(request_timeout);

        metrics_config_json_["metrics_pulsar_broker_address"] = json::value::string(metrics_pulsar_broker_address);
        metrics_config_json_["metrics_pulsar_topic_name"] = json::value::string(metrics_pulsar_topic_name);
        metrics_config_json_["metrics_pulsar_message_key"] = json::value::string(metrics_pulsar_message_key);
        metrics_config_json_["metrics_pulsar_report_period"] = json::value::number(metrics_pulsar_report_period);
        metrics_config_json_["metrics_http_host"] = json::value::string(metrics_http_host);
        metrics_config_json_["metrics_http_port"] = json::value::number(metrics_http_port);

        return true;
    } else {
        return false;
    }
}

json::value diffraflow::MonConfig::collect_metrics() {
    json::value root_json;
    if (zookeeper_setting_ready_flag_) {
        root_json = DynamicConfiguration::collect_metrics();
    }

    root_json["static_config"] = static_config_json_;

    {
        lock_guard<mutex> dynamic_config_json_lg(dynamic_config_json_mtx_);
        root_json["dynamic_config"] = dynamic_config_json_;
    }

    root_json["metrics_config"] = metrics_config_json_;

    return root_json;
}

bool diffraflow::MonConfig::zookeeper_setting_is_ready() { return zookeeper_setting_ready_flag_; }

void diffraflow::MonConfig::print() {
    // with all locks
    lock_guard<mutex> dy_param_string_lg(dy_param_string_mtx_);

    if (zookeeper_setting_ready_flag_) {
        zookeeper_print_setting();
    }

    cout << "static config:" << endl;
    cout << "- node_name = " << node_name << endl;
    cout << "- monitor_id = " << monitor_id << endl;
    cout << "- image_http_host = " << image_http_host << endl;
    cout << "- image_http_port = " << image_http_port << endl;
    cout << "dynamic parameters:" << endl;
    cout << "- dy_param_int = " << dy_param_int_.load() << endl;
    cout << "- dy_param_double = " << dy_param_double_.load() << endl;
    cout << "- dy_param_string = " << dy_param_string_ << endl;
}

bool diffraflow::MonConfig::check_and_commit_(const map<string, string>& conf_map, const time_t conf_mtime) {

    // with all locks
    lock_guard<mutex> dy_param_string_lg(dy_param_string_mtx_);

    // values before commit
    int tmp_dy_param_int = dy_param_int_.load();
    double tmp_dy_param_double = dy_param_double_.load();
    string tmp_dy_param_string = dy_param_string_;

    // convert
    for (map<string, string>::const_iterator iter = conf_map.begin(); iter != conf_map.end(); ++iter) {
        string key = iter->first;
        string value = iter->second;
        if (key == "dy_param_int") {
            tmp_dy_param_int = atoi(value.c_str());
        } else if (key == "dy_param_double") {
            tmp_dy_param_double = atof(value.c_str());
        } else if (key == "dy_param_string") {
            tmp_dy_param_string = value;
        }
    }

    // validation check
    bool invalid_flag = false;
    if (tmp_dy_param_int < 10) {
        cout << "invalid configuration: dy_param_int(" << tmp_dy_param_int << ") is out of range [10, inf)." << endl;
        invalid_flag = true;
    }
    if (tmp_dy_param_double > 1000) {
        cout << "invalid configuration: dy_param_double(" << tmp_dy_param_double << ") is out of range (-inf, 1000]."
             << endl;
        invalid_flag = true;
    }
    if (tmp_dy_param_string.length() < 2) {
        cout << "invalid configuration: dy_param_string(" << tmp_dy_param_string << ") is too short." << endl;
        invalid_flag = true;
    }

    if (invalid_flag) {
        return false;
    }

    // commit change
    if (dy_param_int_ != tmp_dy_param_int) {
        cout << "configuration changed: dy_param_int [ " << dy_param_int_ << " -> " << tmp_dy_param_int << " ]."
             << endl;
        dy_param_int_ = tmp_dy_param_int;
    }
    if (dy_param_double_ != tmp_dy_param_double) {
        cout << "configuration changed: dy_param_double [ " << dy_param_double_ << " -> " << tmp_dy_param_double
             << " ]." << endl;
        dy_param_double_ = tmp_dy_param_double;
    }
    if (dy_param_string_ != tmp_dy_param_string) {
        cout << "configuration changed: dy_param_string [ " << dy_param_string_ << " -> " << tmp_dy_param_string
             << " ]." << endl;
        dy_param_string_ = tmp_dy_param_string;
    }

    config_mtime_ = conf_mtime;

    lock_guard<mutex> dynamic_config_json_lg(dynamic_config_json_mtx_);
    dynamic_config_json_["dy_param_int"] = json::value::number(dy_param_int_);
    dynamic_config_json_["dy_param_double"] = json::value::number(dy_param_double_);
    dynamic_config_json_["dy_param_string"] = json::value::string(dy_param_string_);
    dynamic_config_json_["config_mtime"] = json::value::string(boost::trim_copy(string(ctime(&config_mtime_))));

    return true;
}

int diffraflow::MonConfig::get_dy_param_int() { return dy_param_int_.load(); }

double diffraflow::MonConfig::get_dy_param_double() { return dy_param_double_.load(); }

string diffraflow::MonConfig::get_dy_param_string() {
    lock_guard<mutex> lg(dy_param_string_mtx_);
    return dy_param_string_;
}

bool diffraflow::MonConfig::metrics_pulsar_params_are_set() {
    return (!metrics_pulsar_broker_address.empty() && !metrics_pulsar_topic_name.empty() &&
            !metrics_pulsar_message_key.empty());
}

bool diffraflow::MonConfig::metrics_http_params_are_set() {
    return (!metrics_http_host.empty() && metrics_http_port > 0);
}
