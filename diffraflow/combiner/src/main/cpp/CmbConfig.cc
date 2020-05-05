#include "CmbConfig.hh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

using std::ifstream;
using std::stringstream;
using std::cout;
using std::cerr;
using std::endl;

log4cxx::LoggerPtr diffraflow::CmbConfig::logger_ = log4cxx::Logger::getLogger("CmbConfig");

diffraflow::CmbConfig::CmbConfig() {
    imgfrm_listen_host = "0.0.0.0";
    imgfrm_listen_port = -1;
    imgdat_listen_host = "0.0.0.0";
    imgdat_listen_port = -1;
    imgdat_queue_capacity = 100;

    metrics_pulsar_report_period = 1000;
    metrics_http_port = -1;
}

diffraflow::CmbConfig::~CmbConfig() {}

bool diffraflow::CmbConfig::load(const char* filename) {
    list<pair<string, string>> conf_KV_list;
    if (!read_conf_KV_list_(filename, conf_KV_list)) {
        LOG4CXX_ERROR(logger_, "Failed to read configuration file: " << filename);
        return false;
    }
    for (list<pair<string, string>>::iterator iter = conf_KV_list.begin(); iter != conf_KV_list.end(); ++iter) {
        string key = iter->first;
        string value = iter->second;
        if (key == "imgfrm_listen_host") {
            imgfrm_listen_host = value;
        } else if (key == "imgfrm_listen_port") {
            imgfrm_listen_port = atoi(value.c_str());
        } else if (key == "imgdat_listen_host") {
            imgdat_listen_host = value;
        } else if (key == "imgdat_listen_port") {
            imgdat_listen_port = atoi(value.c_str());
        } else if (key == "imgdat_queue_capacity") {
            imgdat_queue_capacity = atoi(value.c_str());
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

    // if NODE_NAME is defined, use it as the suffix of pulsar_message_key
    if (!metrics_pulsar_message_key.empty()) {
        const char* node_name = getenv("NODE_NAME");
        if (node_name != NULL) {
            metrics_pulsar_message_key += string(".") + string(node_name);
        }
    }

    // correction
    if (metrics_pulsar_report_period < 500) {
        LOG4CXX_WARN(logger_, "pulsar_report_period < 500, use 500 instead.");
        metrics_pulsar_report_period = 500;
    }

    // check
    bool succ_flag = true;
    if (imgfrm_listen_port < 0) {
        LOG4CXX_ERROR(logger_, "invalid imgfrm_listen_port: " << imgfrm_listen_port);
        succ_flag = false;
    }
    if (imgdat_listen_port < 0) {
        LOG4CXX_ERROR(logger_, "invalid imgdat_listen_port: " << imgdat_listen_port);
        succ_flag = false;
    }
    if (imgdat_queue_capacity < 1 || imgdat_queue_capacity > 10000) {
        LOG4CXX_ERROR(logger_, "imgdat_queue_capacity is out of range " << 1 << "-" << 10000);
        succ_flag = false;
    }

    if (succ_flag) {

        static_config_json_["imgfrm_listen_host"] = json::value::string(imgfrm_listen_host);
        static_config_json_["imgfrm_listen_port"] = json::value::number(imgfrm_listen_port);
        static_config_json_["imgdat_listen_host"] = json::value::string(imgdat_listen_host);
        static_config_json_["imgdat_listen_port"] = json::value::number(imgdat_listen_port);
        static_config_json_["imgdat_queue_capacity"] = json::value::number((uint32_t)imgdat_queue_capacity);

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

void diffraflow::CmbConfig::print() {
    cout << " = Configuration Dump Begin =" << endl;
    cout << "  imgfrm_listen_host = " << imgfrm_listen_host << endl;
    cout << "  imgfrm_listen_port = " << imgfrm_listen_port << endl;
    cout << "  imgdat_listen_host = " << imgdat_listen_host << endl;
    cout << "  imgdat_listen_port = " << imgdat_listen_port << endl;
    cout << "  imgdat_queue_capacity = " << imgdat_queue_capacity << endl;
    cout << "  metrics_pulsar_broker_address = " << metrics_pulsar_broker_address << endl;
    cout << "  metrics_pulsar_topic_name = " << metrics_pulsar_topic_name << endl;
    cout << "  metrics_pulsar_message_key = " << metrics_pulsar_message_key << endl;
    cout << "  metrics_pulsar_report_period = " << metrics_pulsar_report_period << endl;
    cout << "  metrics_http_host = " << metrics_http_host << endl;
    cout << "  metrics_http_port = " << metrics_http_port << endl;
    cout << " = Configuration Dump End =" << endl;
}

json::value diffraflow::CmbConfig::collect_metrics() {
    json::value config_json;
    config_json["static_config"] = static_config_json_;
    config_json["metrics_config"] = metrics_config_json_;
    return config_json;
}

bool diffraflow::CmbConfig::metrics_pulsar_params_are_set() {
    return (!metrics_pulsar_broker_address.empty() && !metrics_pulsar_topic_name.empty() &&
            !metrics_pulsar_message_key.empty());
}

bool diffraflow::CmbConfig::metrics_http_params_are_set() {
    return (!metrics_http_host.empty() && metrics_http_port > 0);
}
