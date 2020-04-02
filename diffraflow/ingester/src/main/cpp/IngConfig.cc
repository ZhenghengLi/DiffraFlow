#include "IngConfig.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <ctime>
#include <thread>

using std::cout;
using std::endl;
using std::lock_guard;

log4cxx::LoggerPtr diffraflow::IngConfig::logger_
    = log4cxx::Logger::getLogger("IngConfig");

diffraflow::IngConfig::IngConfig() {
    ingester_id = -1;
    combiner_port = -1;

    combiner_host = "localhost";
    conf_map_in_use["combiner_host"] = "localhost";

}

diffraflow::IngConfig::~IngConfig() {

}

bool diffraflow::IngConfig::load(const char* filename) {
    list< pair<string, string> > conf_KV_list;
    if (!read_conf_KV_list_(filename, conf_KV_list)) {
        LOG4CXX_ERROR(logger_, "Failed to read configuration file: " << filename);
        return false;
    }

    if (zookeeper_parse_setting_(conf_KV_list)) {
        // zookeeper_setting is ready
    }

    lock_guard<mutex> lk(conf_map_mtx_);

    // parse
    for (list< pair<string, string> >::iterator iter = conf_KV_list.begin(); iter != conf_KV_list.end(); ++iter) {
        string key = iter->first;
        string value = iter->second;
        // for static parameters
        if (key == "ingester_id") {
            ingester_id = atoi(value.c_str());
            conf_map_in_use[key] = value;
        } else if (key == "combiner_host") {
            combiner_host = value;
            conf_map_in_use[key] = value;
        } else if (key == "combiner_port") {
            combiner_port = atoi(value.c_str());
            conf_map_in_use[key] = value;
        // for dynamic parameters
        } else if (key == "dy_param_int") {
            dy_param_int = atoi(value.c_str());
            conf_map_in_use[key] = value;
        } else if (key == "dy_param_double") {
            dy_param_double = atof(value.c_str());
            conf_map_in_use[key] = value;
        } else if (key == "dy_param_string") {
            dy_param_string = value;
            conf_map_in_use[key] = value;
        } else {
            LOG4CXX_WARN(logger_, "Found unknown configuration which is ignored: "
                << key << " = " << value << " in " << filename);
        }
    }

    // check
    bool succ_flag = true;
    if (ingester_id < 0) {
        LOG4CXX_ERROR(logger_, "invalid ingester_id: " << ingester_id);
        succ_flag = false;
    }
    if (combiner_port < 0) {
        LOG4CXX_ERROR(logger_, "invalid combiner_port: " << combiner_port);
        succ_flag = false;
    }

    conf_map_mtime_ = time(NULL);
    conf_map_in_use["config_mtime"] = string(ctime(&conf_map_mtime_));

    return succ_flag;
}

void diffraflow::IngConfig::print() {
    zookeeper_print_setting();
    cout << "server config:" << endl;
    cout << "- ingester_id = " << ingester_id << endl;
    cout << "- combiner_host = " << combiner_host << endl;
    cout << "- combiner_port = " << combiner_port << endl;
    cout << "dynamic parameters:" << endl;
    cout << "- dy_param_int = " << dy_param_int << endl;
    cout << "- dy_param_double = " << dy_param_double << endl;
    cout << "- dy_param_string = " << dy_param_string << endl;
}

void diffraflow::IngConfig::convert_and_check_() {
    time_t now_time = time(NULL);
    string now_time_string = boost::trim_copy(string(ctime(&now_time)));
    string mtime_string = boost::trim_copy(string(ctime(&conf_map_mtime_)));
    cout << "configuration updating (" << now_time_string
         << "): received new config map with mtime " << mtime_string << endl;
    for (map<string, string>::iterator iter = conf_map_.begin(); iter != conf_map_.end(); ++iter) {
        if (iter->first == "dy_param_int") {
            int new_value = atoi(iter->second.c_str());
            if (new_value > 10) {
                int old_value = dy_param_int;
                dy_param_int = new_value;
                conf_map_in_use[iter->first] = iter->second;
                cout << "configuration updated  (" << now_time_string
                     << "): dy_param_int " << old_value << " -> " << new_value << endl;
            } else {
                cout << "configuration updating (" << now_time_string
                     << "): dy_param_int " << new_value << " is out of range." << endl;
            }
        } else if (iter->first == "dy_param_double") {
            double new_value = atof(iter->second.c_str());
            if (new_value < 1000) {
                double old_value = dy_param_double;
                dy_param_double = new_value;
                conf_map_in_use[iter->first] = iter->second;
                cout << "configuration updated  (" << now_time_string
                     << "): dy_param_double " << old_value << " -> " << new_value << endl;
            } else {
                cout << "configuration updating (" << now_time_string
                     << "): dy_param_double " << new_value << " is out of range." << endl;
            }
        } else if (iter->first == "dy_param_string") {
            string new_value = iter->second;
            if (new_value.length() > 2) {
                string old_value = dy_param_string;
                dy_param_string = new_value;
                conf_map_in_use[iter->first] = iter->second;
                cout << "configuration updated  (" << now_time_string
                     << "): dy_param_string \"" << old_value << "\" -> \"" << new_value << "\"" << endl;
            } else {
                cout << "configuration updating (" << now_time_string
                     << "): dy_param_string \"" << new_value << "\" is too short." << endl;
            }
        }
    }
    conf_map_in_use["config_mtime"] = mtime_string;
}
