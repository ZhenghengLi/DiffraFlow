#include "IngConfig.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <iostream>
#include <ctime>

using std::cout;
using std::endl;

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
    if (!DynamicConfiguration::load(filename)) {
        return true;
    }

    for (map<string, string>::iterator iter = conf_map_.begin(); iter != conf_map_.end(); ++iter) {
        // for static parameters
        if (iter->first == "ingester_id") {
            ingester_id = atoi(iter->second.c_str());
            conf_map_in_use[iter->first] = iter->second;
            iter = conf_map_.erase(iter);
        } else if (iter->first == "combiner_host") {
            combiner_host = iter->second;
            conf_map_in_use[iter->first] = iter->second;
            iter = conf_map_.erase(iter);
        } else if (iter->first == "combiner_port") {
            combiner_port = atoi(iter->second.c_str());
            conf_map_in_use[iter->first] = iter->second;
            iter = conf_map_.erase(iter);
        }
        // for dynamic parameters
        if (iter->first == "dy_param_int") {
            dy_param_int = atoi(iter->second.c_str());
            conf_map_in_use[iter->first] = iter->second;
            iter = conf_map_.erase(iter);
        } else if (iter->first == "dy_param_double") {
            dy_param_double = atof(iter->second.c_str());
            conf_map_in_use[iter->first] = iter->second;
            iter = conf_map_.erase(iter);
        } else if (iter->first == "dy_param_string") {
            dy_param_string = iter->second;
            conf_map_in_use[iter->first] = iter->second;
            iter = conf_map_.erase(iter);
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
    cout << "configuration updating (" << ctime(&now_time)
         << "): received new config map with mtime " << ctime(&conf_map_mtime_) << endl;
    for (map<string, string>::iterator iter = conf_map_.begin(); iter != conf_map_.end(); ++iter) {
        if (iter->first == "dy_param_int") {
            int new_value = atoi(iter->second.c_str());
            if (new_value > 10) {
                int old_value = dy_param_int;
                dy_param_int = new_value;
                conf_map_in_use[iter->first] = iter->second;
                iter = conf_map_.erase(iter);
                cout << "configuration updated (" << ctime(&now_time)
                     << "): dy_param_int " << old_value << " -> " << new_value << endl;
            } else {
                cout << "configuration updating (" << ctime(&now_time)
                     << "): dy_param_int " << new_value << " is out of range." << endl;
            }
        } else if (iter->first == "dy_param_double") {
            double new_value = atof(iter->second.c_str());
            if (new_value < 1000) {
                double old_value = dy_param_double;
                dy_param_double = new_value;
                conf_map_in_use[iter->first] = iter->second;
                iter = conf_map_.erase(iter);
                cout << "configuration updated (" << ctime(&now_time)
                     << "): dy_param_double " << old_value << " -> " << new_value << endl;
            } else {
                cout << "configuration updating (" << ctime(&now_time)
                     << "): dy_param_double " << new_value << " is out of range." << endl;
            }
        } else if (iter->first == "dy_param_string") {
            string new_value = iter->second;
            if (new_value.length() < 3) {
                string old_value = dy_param_string;
                dy_param_string = new_value;
                conf_map_in_use[iter->first] = iter->second;
                iter = conf_map_.erase(iter);
                cout << "configuration updated (" << ctime(&now_time)
                     << "): dy_param_string \"" << old_value << "\" -> \"" << new_value << "\"" << endl;
            } else {
                cout << "configuration updating (" << ctime(&now_time)
                     << "): dy_param_string " << new_value << " is too short." << endl;
            }
        }
    }
    conf_map_in_use["config_mtime"] = string(ctime(&conf_map_mtime_));
}
