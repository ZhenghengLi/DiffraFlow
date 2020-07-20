#include "SndConfig.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <regex>
#include <boost/algorithm/string.hpp>

using std::cout;
using std::flush;
using std::endl;
using std::regex;
using std::regex_match;
using std::regex_replace;

log4cxx::LoggerPtr diffraflow::SndConfig::logger_ = log4cxx::Logger::getLogger("SndConfig");

diffraflow::SndConfig::SndConfig() {
    sender_id = 0;
    listen_host = "0.0.0.0";
    listen_port = -1;

    data_dir.clear();
    max_seq_num = 0;
    events_per_file = 10000;
    total_events = 89000;

    dispatcher_host = "localhost";
    dispatcher_port = -1;
    module_id = -1;
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
        } else if (key == "sender_id") {
            sender_id = atoi(value.c_str());
        } else if (key == "data_dir") {
            data_dir = value.c_str();
        } else if (key == "max_seq_num") {
            max_seq_num = atoi(value.c_str());
        } else if (key == "events_per_file") {
            events_per_file = atoi(value.c_str());
        } else if (key == "total_events") {
            total_events = atoi(value.c_str());
        } else if (key == "dispatcher_host") {
            dispatcher_host = value.c_str();
        } else if (key == "dispatcher_port") {
            dispatcher_port = atoi(value.c_str());
        } else if (key == "module_id") {
            module_id = atoi(value.c_str());
        } else {
            LOG4CXX_WARN(logger_,
                "Found unknown configuration which is ignored: " << key << " = " << value << " in " << filename);
        }
    }

    // const char* node_name = getenv("NODE_NAME");

    // use POD IP as dispatcher_id
    if (sender_id == 0) {
        const char* pod_ip = getenv("POD_IP");
        if (pod_ip != NULL) {
            vector<string> ip_nums;
            boost::split(ip_nums, pod_ip, boost::is_any_of("."));
            for (size_t i = 0; i < ip_nums.size(); i++) {
                sender_id <<= 8;
                sender_id += atoi(ip_nums[i].c_str());
            }
        }
    }

    // check
    bool succ_flag = true;
    if (listen_port < 0) {
        LOG4CXX_ERROR(logger_, "invalid listen_port: " << listen_port);
        succ_flag = false;
    }

    if (data_dir.empty()) {
        LOG4CXX_ERROR(logger_, "data_dir is not set.")
        succ_flag = false;
    }

    if (max_seq_num < 0) {
        LOG4CXX_ERROR(logger_, "max_seq_num < 0.");
        succ_flag = false;
    }
    if (events_per_file < 0) {
        LOG4CXX_ERROR(logger_, "events_per_file < 0.");
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

    return succ_flag;
}

bool diffraflow::SndConfig::load_nodemap(const char* filename) { return true; }

void diffraflow::SndConfig::print() {
    cout << " ---- Configuration Dump Begin ----" << endl;
    cout << " sender_id          = " << sender_id << endl;
    cout << " listen_host        = " << listen_host << endl;
    cout << " listen_port        = " << listen_port << endl;
    cout << " data_dir           = " << data_dir << endl;
    cout << " max_seq_num        = " << max_seq_num << endl;
    cout << " events_per_file    = " << events_per_file << endl;
    cout << " total_events       = " << total_events << endl;
    cout << " dispatcher_host    = " << dispatcher_host << endl;
    cout << " dispatcher_port    = " << dispatcher_port << endl;
    cout << " module_id          = " << module_id << endl;
    cout << " ---- Configuration Dump End ----" << endl;
}
