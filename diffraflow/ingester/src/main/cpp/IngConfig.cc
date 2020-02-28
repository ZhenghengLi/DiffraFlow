#include "IngConfig.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <iostream>

using std::cout;
using std::endl;

log4cxx::LoggerPtr diffraflow::IngConfig::logger_
    = log4cxx::Logger::getLogger("IngConfig");

diffraflow::IngConfig::IngConfig() {
    ingester_id = -1;
    combiner_host = "localhost";
    combiner_port = -1;

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

    return succ_flag;
}

void diffraflow::IngConfig::print() {

}

void diffraflow::IngConfig::convert_and_check_() {
    for (map<string, string>::iterator iter = conf_map_.begin(); iter != conf_map_.end(); ++iter) {
        if (iter->first == "dy_param_int") {
            int value = atoi(iter->second.c_str());
            if (value > 10) {
                dy_param_int = value;
                conf_map_in_use[iter->first] = iter->second;
                iter = conf_map_.erase(iter);
            }
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
}