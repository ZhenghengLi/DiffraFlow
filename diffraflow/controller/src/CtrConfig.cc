#include "CtrConfig.hh"
#include <iostream>
#include <cstdlib>
#include <regex>

using std::cout;
using std::endl;
using std::flush;
using std::regex;
using std::regex_match;
using std::regex_replace;

log4cxx::LoggerPtr diffraflow::CtrConfig::logger_ = log4cxx::Logger::getLogger("CtrConfig");

diffraflow::CtrConfig::CtrConfig() {
    request_timeout = 10000; // 10 seconds
    http_port = -1;

    metrics_pulsar_report_period = 1000;
    metrics_http_port = -1;
}

diffraflow::CtrConfig::~CtrConfig() {}

bool diffraflow::CtrConfig::load(const char* filename) {
    list<pair<string, string>> conf_KV_list;
    if (!read_conf_KV_list_(filename, conf_KV_list)) {
        LOG4CXX_ERROR(logger_, "Failed to read configuration file: " << filename);
        return false;
    }
    // parse
    for (list<pair<string, string>>::iterator iter = conf_KV_list.begin(); iter != conf_KV_list.end(); ++iter) {
        string key = iter->first;
        string value = iter->second;
        if (key == "http_host") {
            http_host = value;
        } else if (key == "http_port") {
            http_port = atoi(value.c_str());
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
        } else {
            LOG4CXX_WARN(logger_,
                "Found unknown configuration which is ignored: " << key << " = " << value << " in " << filename);
        }
    }

    // replace NODE_NAME in message key
    const char* node_name_cstr = getenv("NODE_NAME");
    if (node_name_cstr != nullptr && regex_match(metrics_pulsar_message_key, regex(".*NODE_NAME.*"))) {
        metrics_pulsar_message_key = regex_replace(metrics_pulsar_message_key, regex("NODE_NAME"), node_name_cstr);
    }

    // currection
    if (metrics_pulsar_report_period < 500) {
        LOG4CXX_WARN(logger_, "pulsar_report_period < 500, use 500 instead.");
        metrics_pulsar_report_period = 500;
    }
    if (request_timeout < 100) {
        LOG4CXX_WARN(logger_, "request_timeout is too small (< 100), use 100 instead.");
        request_timeout = 100;
    }
    // check
    bool succ_flag = true;
    if (http_host.empty()) {
        LOG4CXX_ERROR(logger_, "http_host is not set.");
        succ_flag = false;
    }
    if (http_port < 0) {
        LOG4CXX_ERROR(logger_, "invalid http_port: " << http_port);
        succ_flag = false;
    }

    if (succ_flag) {
        static_config_json_["http_host"] = json::value::string(http_host);
        static_config_json_["http_port"] = json::value::number(http_port);
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

void diffraflow::CtrConfig::print() {
    cout << "Configuration:" << endl;
    cout << "- http_host = " << http_host << endl;
    cout << "- http_port = " << http_port << endl;
    cout << "- request_timeout = " << request_timeout << endl;
    cout << "  metrics_pulsar_broker_address = " << metrics_pulsar_broker_address << endl;
    cout << "  metrics_pulsar_topic_name = " << metrics_pulsar_topic_name << endl;
    cout << "  metrics_pulsar_message_key = " << metrics_pulsar_message_key << endl;
    cout << "  metrics_pulsar_report_period = " << metrics_pulsar_report_period << endl;
    cout << "  metrics_http_host = " << metrics_http_host << endl;
    cout << "  metrics_http_port = " << metrics_http_port << endl;
}

json::value diffraflow::CtrConfig::collect_metrics() {
    json::value config_json;
    config_json["static_config"] = static_config_json_;
    config_json["metrics_config"] = metrics_config_json_;
    return config_json;
}

bool diffraflow::CtrConfig::metrics_pulsar_params_are_set() {
    return (!metrics_pulsar_broker_address.empty() && !metrics_pulsar_topic_name.empty() &&
            !metrics_pulsar_message_key.empty());
}

bool diffraflow::CtrConfig::metrics_http_params_are_set() {
    return (!metrics_http_host.empty() && metrics_http_port > 0);
}
