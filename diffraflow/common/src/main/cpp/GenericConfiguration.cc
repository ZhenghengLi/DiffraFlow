#include "GenericConfiguration.hh"
#include <iostream>
#include <fstream>
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <boost/algorithm/string.hpp>

using std::ifstream;

log4cxx::LoggerPtr diffraflow::GenericConfiguration::logger_ = log4cxx::Logger::getLogger("GenericConfiguration");

diffraflow::GenericConfiguration::GenericConfiguration() {}

diffraflow::GenericConfiguration::~GenericConfiguration() {}

bool diffraflow::GenericConfiguration::read_conf_KV_list_(
    const char* filename, list<pair<string, string>>& conf_KV_list) {
    conf_KV_list.clear();
    ifstream conf_file;
    conf_file.open(filename);
    if (!conf_file.is_open()) {
        LOG4CXX_ERROR(logger_, "configuration file open failed.");
        return false;
    }
    string oneline;
    while (true) {
        oneline = "";
        getline(conf_file, oneline);
        if (conf_file.eof()) break;
        // skip comments
        boost::trim(oneline);
        if (oneline[0] == '#') continue;
        if (oneline.length() == 0) continue;
        // extract host and port
        vector<string> key_value;
        boost::split(key_value, oneline, boost::is_any_of("="));
        if (key_value.size() < 2) {
            LOG4CXX_ERROR(logger_, "found unknown configuration: " << oneline);
            conf_file.close();
            return false;
        }
        boost::trim(key_value[0]);
        boost::trim(key_value[1]);
        if (key_value[0].size() > 0 && key_value[1].size() > 0) {
            conf_KV_list.push_back(make_pair(key_value[0], key_value[1]));
        }
    }
    conf_file.close();
    if (conf_KV_list.size() > 0) {
        return true;
    } else {
        LOG4CXX_WARN(logger_, "there is no valid configurations in file: " << filename);
        return false;
    }
}
