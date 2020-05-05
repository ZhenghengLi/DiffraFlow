#include "DspConfig.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <boost/algorithm/string.hpp>

using std::cout;
using std::flush;
using std::endl;

log4cxx::LoggerPtr diffraflow::DspConfig::logger_ = log4cxx::Logger::getLogger("DspConfig");

diffraflow::DspConfig::DspConfig() {
    dispatcher_id = 0;
    listen_host = "0.0.0.0";
    listen_port = -1;
    compress_method = DspSender::kNone;

    metrics_pulsar_report_period = 1000;
    metrics_http_port = -1;
}

diffraflow::DspConfig::~DspConfig() {}

bool diffraflow::DspConfig::load(const char* filename) {
    list<pair<string, string>> conf_KV_list;
    if (!read_conf_KV_list_(filename, conf_KV_list)) {
        LOG4CXX_ERROR(logger_, "Failed to read configuration file: " << filename);
        return false;
    }
    // parse
    for (list<pair<string, string>>::iterator iter = conf_KV_list.begin(); iter != conf_KV_list.end(); ++iter) {
        string key = iter->first;
        string value = iter->second;
        if (key == "listen_host") {
            listen_host = value;
        } else if (key == "listen_port") {
            listen_port = atoi(value.c_str());
        } else if (key == "dispatcher_id") {
            dispatcher_id = atoi(value.c_str());
        } else if (key == "compress_method") {
            if (value == "LZ4") {
                compress_method = DspSender::kLZ4;
            } else if (value == "Snappy") {
                compress_method = DspSender::kSnappy;
            } else if (value == "ZSTD") {
                compress_method = DspSender::kZSTD;
            } else {
                compress_method = DspSender::kNone;
            }
        } else if (key == "compress_level") {
            compress_level = atoi(value.c_str());
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

    // use POD IP as dispatcher_id
    if (dispatcher_id == 0) {
        const char* pod_ip = getenv("POD_IP");
        if (pod_ip != NULL) {
            vector<string> ip_nums;
            boost::split(ip_nums, pod_ip, boost::is_any_of("."));
            for (size_t i = 0; i < ip_nums.size(); i++) {
                dispatcher_id <<= 8;
                dispatcher_id += atoi(ip_nums[i].c_str());
            }
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
    if (listen_port < 0) {
        LOG4CXX_ERROR(logger_, "invalid listen_port: " << listen_port);
        succ_flag = false;
    }
    if (compress_method == DspSender::kZSTD && (compress_level < 1 || compress_level >= 20)) {
        LOG4CXX_ERROR(logger_, "compress level for ZSTD compress method is out of range, it should be >= 1 and < 20.");
        succ_flag = false;
    }

    if (succ_flag) {
        static_config_json_["dispatcher_id"] = json::value::number(dispatcher_id);
        static_config_json_["listen_host"] = json::value::string(listen_host);
        static_config_json_["listen_port"] = json::value::number(listen_port);
        switch (compress_method) {
        case DspSender::kLZ4:
            static_config_json_["compress_method"] = json::value::string("LZ4");
            break;
        case DspSender::kSnappy:
            static_config_json_["compress_method"] = json::value::string("Snappy");
            break;
        case DspSender::kZSTD:
            static_config_json_["compress_method"] = json::value::string("ZSTD");
            break;
        default:
            static_config_json_["compress_method"] = json::value::string("None");
        }
        static_config_json_["compress_level"] = json::value::number(compress_level);

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

void diffraflow::DspConfig::print() {
    cout << " ---- Configuration Dump Begin ----" << endl;
    cout << "  listen_port = " << listen_port << endl;
    cout << "  dispatcher_id = " << dispatcher_id << endl;
    cout << "  compress_method = " << flush;
    switch (compress_method) {
    case DspSender::kLZ4:
        cout << "LZ4" << endl;
        break;
    case DspSender::kSnappy:
        cout << "Snappy" << endl;
        break;
    case DspSender::kZSTD:
        cout << "ZSTD (" << compress_level << ")" << endl;
        break;
    default:
        cout << "None" << endl;
    }
    cout << "  compress_level = " << compress_level << endl;
    cout << "  metrics_pulsar_broker_address = " << metrics_pulsar_broker_address << endl;
    cout << "  metrics_pulsar_topic_name = " << metrics_pulsar_topic_name << endl;
    cout << "  metrics_pulsar_message_key = " << metrics_pulsar_message_key << endl;
    cout << "  metrics_pulsar_report_period = " << metrics_pulsar_report_period << endl;
    cout << "  metrics_http_host = " << metrics_http_host << endl;
    cout << "  metrics_http_port = " << metrics_http_port << endl;
    cout << " ---- Configuration Dump End ----" << endl;
}

json::value diffraflow::DspConfig::collect_metrics() {
    json::value config_json;
    config_json["static_config"] = static_config_json_;
    config_json["metrics_config"] = metrics_config_json_;
    return config_json;
}

bool diffraflow::DspConfig::metrics_pulsar_params_are_set() {
    return (!metrics_pulsar_broker_address.empty() && !metrics_pulsar_topic_name.empty() &&
            !metrics_pulsar_message_key.empty());
}

bool diffraflow::DspConfig::metrics_http_params_are_set() {
    return (!metrics_http_host.empty() && metrics_http_port > 0);
}
