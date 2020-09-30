#include "AggConfig.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <regex>
#include <thread>
#include <boost/algorithm/string.hpp>

using std::cout;
using std::flush;
using std::endl;
using std::regex;
using std::regex_match;
using std::regex_replace;

log4cxx::LoggerPtr diffraflow::AggConfig::logger_ = log4cxx::Logger::getLogger("AggConfig");

diffraflow::AggConfig::AggConfig() {
    http_server_host = "0.0.0.0";
    http_server_port = -1;

    metrics_pulsar_report_period = 1000;
    metrics_http_port = -1;
}

diffraflow::AggConfig::~AggConfig() {}

bool diffraflow::AggConfig::load(const char* filename) {
    list<pair<string, string>> conf_KV_list;
    if (!read_conf_KV_list_(filename, conf_KV_list)) {
        LOG4CXX_ERROR(logger_, "Failed to read configuration file: " << filename);
        return false;
    }
    // parse
    for (list<pair<string, string>>::iterator iter = conf_KV_list.begin(); iter != conf_KV_list.end(); ++iter) {
        string key = iter->first;
        string value = iter->second;
        if (key == "http_server_host") {
            http_server_host = value;
        } else if (key == "http_server_port") {
            http_server_port = atoi(value.c_str());
        } else if (key == "pulsar_url") {
            pulsar_url = value;
        } else if (key == "controller_topic") {
            controller_topic = value;
        } else if (key == "sender_topic") {
            sender_topic = value;
        } else if (key == "dispatcher_topic") {
            dispatcher_topic = value;
        } else if (key == "combiner_topic") {
            combiner_topic = value;
        } else if (key == "ingester_topic") {
            ingester_topic = value;
        } else if (key == "monitor_topic") {
            monitor_topic = value;
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

    // check
    bool succ_flag = true;
    if (http_server_port < 0) {
        LOG4CXX_ERROR(logger_, "invalid http_server_port: " << http_server_port);
        succ_flag = false;
    }
    if (pulsar_url.empty()) {
        LOG4CXX_ERROR(logger_, "pulsr_url is not set.");
        succ_flag = false;
    }

    if (succ_flag) {
        static_config_json_["http_server_host"] = json::value::string(http_server_host);
        static_config_json_["http_server_port"] = json::value::number(http_server_port);
        static_config_json_["pulsar_url"] = json::value::string(pulsar_url);
        static_config_json_["controller_topic"] = json::value::string(controller_topic);
        static_config_json_["sender_topic"] = json::value::string(sender_topic);
        static_config_json_["dispatcher_topic"] = json::value::string(dispatcher_topic);
        static_config_json_["combiner_topic"] = json::value::string(combiner_topic);
        static_config_json_["ingester_topic"] = json::value::string(ingester_topic);
        static_config_json_["monitor_topic"] = json::value::string(monitor_topic);

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

void diffraflow::AggConfig::print() {
    cout << " ---- Configuration Dump Begin ----" << endl;
    cout << "  http_server_host = " << http_server_host << endl;
    cout << "  http_server_port = " << http_server_port << endl;
    cout << "  pulsar_url = " << pulsar_url << endl;
    cout << "  controller_topic = " << controller_topic << endl;
    cout << "  sender_topic = " << sender_topic << endl;
    cout << "  dispatcher_topic = " << dispatcher_topic << endl;
    cout << "  combiner_topic = " << combiner_topic << endl;
    cout << "  ingester_topic = " << ingester_topic << endl;
    cout << "  monitor_topic = " << monitor_topic << endl;
    cout << "  metrics_pulsar_broker_address = " << metrics_pulsar_broker_address << endl;
    cout << "  metrics_pulsar_topic_name = " << metrics_pulsar_topic_name << endl;
    cout << "  metrics_pulsar_message_key = " << metrics_pulsar_message_key << endl;
    cout << "  metrics_pulsar_report_period = " << metrics_pulsar_report_period << endl;
    cout << "  metrics_http_host = " << metrics_http_host << endl;
    cout << "  metrics_http_port = " << metrics_http_port << endl;
    cout << " ---- Configuration Dump End ----" << endl;
}

json::value diffraflow::AggConfig::collect_metrics() {
    json::value config_json;
    config_json["static_config"] = static_config_json_;
    config_json["metrics_config"] = metrics_config_json_;
    return config_json;
}

bool diffraflow::AggConfig::metrics_pulsar_params_are_set() {
    return (!metrics_pulsar_broker_address.empty() && !metrics_pulsar_topic_name.empty() &&
            !metrics_pulsar_message_key.empty());
}

bool diffraflow::AggConfig::metrics_http_params_are_set() {
    return (!metrics_http_host.empty() && metrics_http_port > 0);
}
