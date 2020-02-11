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

log4cxx::LoggerPtr diffraflow::CmbConfig::logger_
    = log4cxx::Logger::getLogger("CmbConfig");

diffraflow::CmbConfig::CmbConfig() {

}

diffraflow::CmbConfig::~CmbConfig() {

}

bool diffraflow::CmbConfig::load(const char* filename) {
    vector< pair<string, string> > conf_KV_vec;
    if (!read_conf_KV_vec_(filename, conf_KV_vec)) {
        LOG4CXX_ERROR(logger_, "Failed to read configuration file: " << filename);
        return false;
    }
    for (size_t i = 0; i < conf_KV_vec.size(); i++) {
        string key = conf_KV_vec[i].first;
        string value = conf_KV_vec[i].second;
        if (key == "listen_host") {
            listen_host = value;
        } else if (key == "listen_port") {
            listen_port = atoi(value.c_str());
        } else {
            LOG4CXX_WARN(logger_, "Found unknown configuration which is ignored: "
                << key << " = " << value << " in " << filename);
        }
    }
    // check
    bool succ_flag = true;
    if (listen_port < 0) {
        LOG4CXX_ERROR(logger_, "invalid listen_port: " << listen_port);
        succ_flag = false;
    }
    return succ_flag;
}

void diffraflow::CmbConfig::print() {
    cout << " = Configuration Dump Begin =" << endl;
    cout << "  listen_host = " << listen_host << endl;
    cout << "  listen_port = " << listen_port << endl;
    cout << " = Configuration Dump End =" << endl;

}
