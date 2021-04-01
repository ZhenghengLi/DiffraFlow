#include "DspConfig.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <regex>
#include <thread>
#include <boost/algorithm/string.hpp>

#include "schedtools.hh"

using std::cout;
using std::flush;
using std::endl;
using std::regex;
using std::regex_match;
using std::regex_replace;

log4cxx::LoggerPtr diffraflow::DspConfig::logger_ = log4cxx::Logger::getLogger("DspConfig");

diffraflow::DspConfig::DspConfig() {
    dispatcher_id = 0;
    dgram_recv_cpu_id = -1;
    dgram_recv_buffer_size = 56 * 1024 * 1024;
    listen_host = "0.0.0.0";
    listen_port = -1;
    max_queue_size = 1000;

    other_cpu_list.clear();
    if (sched_getaffinity(0, sizeof(cpu_set_t), &other_cpu_set)) {
        LOG4CXX_WARN(logger_, "Failed to sched_getaffinity: " << strerror(errno));
    }

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
        } else if (key == "dgram_recv_cpu_id") {
            dgram_recv_cpu_id = atoi(value.c_str());
        } else if (key == "dgram_recv_buffer_size") {
            dgram_recv_buffer_size = atoi(value.c_str());
        } else if (key == "max_queue_size") {
            max_queue_size = atoi(value.c_str());
        } else if (key == "other_cpu_list") {
            other_cpu_list = value;
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
            if (ip_nums.size() == 4) {
                dispatcher_id += atoi(ip_nums[2].c_str());
                dispatcher_id <<= 8;
                dispatcher_id += atoi(ip_nums[3].c_str());
                dispatcher_id <<= 16;
                if (listen_port > 0) {
                    dispatcher_id += listen_port;
                }
            }
        }
    }
    // replace the NODE_NAME in pulsar_message_key
    if (!metrics_pulsar_message_key.empty()) {
        const char* node_name = getenv("NODE_NAME");
        if (node_name != NULL && regex_match(metrics_pulsar_message_key, regex(".*NODE_NAME.*"))) {
            metrics_pulsar_message_key = regex_replace(metrics_pulsar_message_key, regex("NODE_NAME"), node_name);
        }
    }

    // correction
    if (max_queue_size < 100) {
        LOG4CXX_WARN(logger_, "max_queue_size is too small (<100), use 100 instead.");
        max_queue_size = 100;
    }
    if (dgram_recv_buffer_size < 512 * 1024) {
        LOG4CXX_WARN(logger_, "dgram_recv_buffer_size it too small (< 512 kiB) use 512 kiB instead.");
        dgram_recv_buffer_size = 512 * 1024;
    } else if (dgram_recv_buffer_size > 128 * 1024 * 1024) {
        LOG4CXX_WARN(logger_, "dgram_recv_buffer_size is too large (> 128 MiB), use 128 MiB instead.");
        dgram_recv_buffer_size = 128 * 1024 * 1024;
    }
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
    int num_cpus = std::thread::hardware_concurrency();
    if (dgram_recv_cpu_id >= num_cpus) {
        LOG4CXX_ERROR(logger_, "dgram_recv_cpu_id should be smaller than " << num_cpus << ".");
        succ_flag = false;
    }
    if (!other_cpu_list.empty()) {
        int res = schedtools::string_to_cpu_set(&other_cpu_set, other_cpu_list);
        if (res != 0) {
            LOG4CXX_ERROR(
                logger_, "failed to convert string '" << other_cpu_list << "' to cpu_set with error code: " << res);
            succ_flag = false;
        }
    }

    if (succ_flag) {
        static_config_json_["dispatcher_id"] = json::value::number(dispatcher_id);
        static_config_json_["listen_host"] = json::value::string(listen_host);
        static_config_json_["listen_port"] = json::value::number(listen_port);
        static_config_json_["max_queue_size"] = json::value::number(max_queue_size);
        static_config_json_["other_cpu_list"] = json::value::string(other_cpu_list);
        static_config_json_["dgram_recv_cpu_id"] = json::value::number(dgram_recv_cpu_id);
        static_config_json_["dgram_recv_buffer_size"] = json::value::number(dgram_recv_buffer_size);

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
    cout << "  dgram_recv_cpu_id = " << dgram_recv_cpu_id << endl;
    cout << "  dgram_recv_buffer_size = " << dgram_recv_buffer_size << endl;
    cout << "  max_queue_size = " << max_queue_size << endl;
    cout << "  other_cpu_list = " << other_cpu_list << endl;
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
