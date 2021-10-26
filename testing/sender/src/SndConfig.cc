#include "SndConfig.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <regex>
#include <boost/algorithm/string.hpp>
#include <thread>

using std::cout;
using std::flush;
using std::endl;
using std::regex;
using std::regex_match;
using std::regex_replace;
using std::ifstream;

log4cxx::LoggerPtr diffraflow::SndConfig::logger_ = log4cxx::Logger::getLogger("SndConfig");

diffraflow::SndConfig::SndConfig() {
    sender_id = 0;
    sender_cpu_id = -1;
    sender_buffer_size = 4 * 1024 * 1024;
    listen_host = "0.0.0.0";
    listen_port = -1;
    sender_type = "TCP";

    sender_port = -1;

    data_dir.clear();
    events_per_file = 10000;
    start_event = 0;
    total_events = 89000;

    dispatcher_host = "localhost";
    dispatcher_port = -1;
    module_id = -1;

    metrics_pulsar_report_period = 1000;
    metrics_http_port = -1;
}

diffraflow::SndConfig::~SndConfig() {}

bool diffraflow::SndConfig::load(const char* filename) {
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
        } else if (key == "sender_type") {
            sender_type = value.c_str();
        } else if (key == "sender_id") {
            sender_id = atoi(value.c_str());
        } else if (key == "sender_port") {
            sender_port = atoi(value.c_str());
        } else if (key == "sender_cpu_id") {
            sender_cpu_id = atoi(value.c_str());
        } else if (key == "sender_buffer_size") {
            sender_buffer_size = atoi(value.c_str());
        } else if (key == "data_dir") {
            data_dir = value.c_str();
        } else if (key == "events_per_file") {
            events_per_file = atoi(value.c_str());
        } else if (key == "start_event") {
            start_event = atoi(value.c_str());
        } else if (key == "total_events") {
            total_events = atoi(value.c_str());
        } else if (key == "dispatcher_host") {
            dispatcher_host = value.c_str();
        } else if (key == "dispatcher_port") {
            dispatcher_port = atoi(value.c_str());
        } else if (key == "module_id") {
            module_id = atoi(value.c_str());
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
    if (sender_id == 0) {
        const char* pod_ip = getenv("POD_IP");
        if (pod_ip != NULL) {
            vector<string> ip_nums;
            boost::split(ip_nums, pod_ip, boost::is_any_of("."));
            if (ip_nums.size() == 4) {
                sender_id += atoi(ip_nums[2].c_str());
                sender_id <<= 8;
                sender_id += atoi(ip_nums[3].c_str());
                sender_id <<= 16;
                if (listen_port > 0) {
                    sender_id += listen_port;
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
    if (sender_buffer_size < 512 * 1024) {
        LOG4CXX_WARN(logger_, "sender_buffer_size is too small (< 512 kiB), use 512 kiB instead.");
        sender_buffer_size = 512 * 1024;
    } else if (sender_buffer_size > 64 * 1024 * 1024) {
        LOG4CXX_WARN(logger_, "sender_buffer_size is too large (> 64 MiB), use 64 MiB instead.");
        sender_buffer_size = 64 * 1024 * 1024;
    }
    if (metrics_pulsar_report_period < 500) {
        LOG4CXX_WARN(logger_, "pulsar_report_period < 500, use 500 instead.");
        metrics_pulsar_report_period = 500;
    }

    // check
    bool succ_flag = true;
    if (!(sender_type == "TCP" || sender_type == "UDP")) {
        LOG4CXX_ERROR(logger_, "sender_type must be TCP or UDP, currently it is " << sender_type);
        succ_flag = false;
    }
    if (listen_port < 0) {
        LOG4CXX_ERROR(logger_, "invalid listen_port: " << listen_port);
        succ_flag = false;
    }
    int num_cpus = std::thread::hardware_concurrency();
    if (sender_cpu_id >= num_cpus) {
        LOG4CXX_ERROR(logger_, "sender_cpu_id should be smaller than " << num_cpus << ".");
        succ_flag = false;
    }
    if (data_dir.empty()) {
        LOG4CXX_ERROR(logger_, "data_dir is not set.")
        succ_flag = false;
    }
    if (events_per_file < 0) {
        LOG4CXX_ERROR(logger_, "events_per_file < 0.");
        succ_flag = false;
    }
    if (start_event < 0) {
        LOG4CXX_ERROR(logger_, "start_event < 0.");
        succ_flag = false;
    }
    if (total_events < 0) {
        LOG4CXX_ERROR(logger_, "total_events < 0.");
        succ_flag = false;
    }
    if (dispatcher_port < 0) {
        LOG4CXX_ERROR(logger_, "invalid dispatcher_port: " << dispatcher_port);
        succ_flag = false;
    }
    if (module_id < 0 || module_id > 15) {
        LOG4CXX_ERROR(logger_, "module_id is out of range (0 -- 15): " << module_id);
        succ_flag = false;
    }

    if (succ_flag) {
        // static config
        static_config_json_["sender_type"] = json::value::string(sender_type);
        static_config_json_["sender_id"] = json::value::number(sender_id);
        static_config_json_["sender_port"] = json::value::number(sender_port);
        static_config_json_["sender_cpu_id"] = json::value::number(sender_cpu_id);
        static_config_json_["sender_buffer_size"] = json::value::number(sender_buffer_size);
        static_config_json_["listen_host"] = json::value::string(listen_host);
        static_config_json_["listen_port"] = json::value::number(listen_port);
        static_config_json_["data_dir"] = json::value::string(data_dir);
        static_config_json_["events_per_file"] = json::value::number(events_per_file);
        static_config_json_["total_events"] = json::value::number(total_events);
        static_config_json_["dispatcher_host"] = json::value::string(dispatcher_host);
        static_config_json_["dispatcher_port"] = json::value::number(dispatcher_port);
        static_config_json_["module_id"] = json::value::number(module_id);
        // metrics config
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

bool diffraflow::SndConfig::load_nodemap(const char* filename, const string nodename) {
    ifstream nodemap_file;
    nodemap_file.open(filename);
    if (!nodemap_file.is_open()) {
        LOG4CXX_ERROR(logger_, "node map file open failed.");
        return false;
    }
    string oneline;
    bool found_flag = false;
    while (true) {
        oneline = "";
        getline(nodemap_file, oneline);
        if (nodemap_file.eof()) break;
        // skip comments
        boost::trim(oneline);
        if (oneline[0] == '#') continue;
        if (oneline.length() == 0) continue;
        // extract fields formated as "node_name, module_id, dispatcher_host, dispatcher_port"
        vector<string> fields;
        boost::split(fields, oneline, boost::is_any_of(","));
        if (fields.size() < 4) {
            LOG4CXX_ERROR(logger_, "invalid node map line: " << oneline);
            nodemap_file.close();
            return false;
        }
        if (boost::trim_copy(fields[0]) == nodename) {
            module_id = std::stoi(boost::trim_copy(fields[1]));
            dispatcher_host = boost::trim_copy(fields[2]);
            dispatcher_port = std::stoi(boost::trim_copy(fields[3]));
            found_flag = true;
            break;
        }
    }
    nodemap_file.close();
    if (found_flag) {
        return true;
    } else {
        LOG4CXX_ERROR(logger_, "there is no node name " << nodename << " in file " << filename << ".");
        return false;
    }
}

void diffraflow::SndConfig::print() {
    cout << " ---- Configuration Dump Begin ----" << endl;
    cout << " sender_type        = " << sender_type << endl;
    cout << " sender_id          = " << sender_id << endl;
    cout << " sender_port        = " << sender_port << endl;
    cout << " sender_cpu_id      = " << sender_cpu_id << endl;
    cout << " sender_buffer_size = " << sender_buffer_size << endl;
    cout << " listen_host        = " << listen_host << endl;
    cout << " listen_port        = " << listen_port << endl;
    cout << " data_dir           = " << data_dir << endl;
    cout << " events_per_file    = " << events_per_file << endl;
    cout << " total_events       = " << total_events << endl;
    cout << " dispatcher_host    = " << dispatcher_host << endl;
    cout << " dispatcher_port    = " << dispatcher_port << endl;
    cout << " module_id          = " << module_id << endl;
    cout << " metrics_pulsar_broker_address = " << metrics_pulsar_broker_address << endl;
    cout << " metrics_pulsar_topic_name = " << metrics_pulsar_topic_name << endl;
    cout << " metrics_pulsar_message_key = " << metrics_pulsar_message_key << endl;
    cout << " metrics_pulsar_report_period = " << metrics_pulsar_report_period << endl;
    cout << " metrics_http_host = " << metrics_http_host << endl;
    cout << " metrics_http_port = " << metrics_http_port << endl;
    cout << " ---- Configuration Dump End ----" << endl;
}

json::value diffraflow::SndConfig::collect_metrics() {
    json::value config_json;
    config_json["static_config"] = static_config_json_;
    config_json["metrics_config"] = metrics_config_json_;
    return config_json;
}

bool diffraflow::SndConfig::metrics_pulsar_params_are_set() {
    return (!metrics_pulsar_broker_address.empty() && !metrics_pulsar_topic_name.empty() &&
            !metrics_pulsar_message_key.empty());
}

bool diffraflow::SndConfig::metrics_http_params_are_set() {
    return (!metrics_http_host.empty() && metrics_http_port > 0);
}