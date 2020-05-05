#include "CtrConfig.hh"
#include <iostream>

using std::cout;
using std::endl;
using std::flush;

log4cxx::LoggerPtr diffraflow::CtrConfig::logger_ = log4cxx::Logger::getLogger("CtrConfig");

diffraflow::CtrConfig::CtrConfig() {
    request_timeout = 10000; // 10 seconds
    http_port = -1;
}

diffraflow::CtrConfig::~CtrConfig() {}

bool diffraflow::CtrConfig::load(const char* filename) {
    list<pair<string, string>> conf_KV_list;
    if (!read_conf_KV_list_(filename, conf_KV_list)) {
        LOG4CXX_ERROR(logger_, "Failed to read configuration file: " << filename);
        return false;
    }
    // parse
    for (list<pair<string, string>>::iterator iter = conf_KV_list.begin(); iter != conf_KV_list.end(); ++iter) {
        string key = iter->first;
        string value = iter->second;
        if (key == "http_host") {
            http_host = value;
        } else if (key == "http_port") {
            http_port = atoi(value.c_str());
        } else if (key == "request_timeout") {
            request_timeout = atoi(value.c_str());
        } else {
            LOG4CXX_WARN(logger_,
                "Found unknown configuration which is ignored: " << key << " = " << value << " in " << filename);
        }
    }
    // currection
    if (request_timeout < 100) {
        LOG4CXX_WARN(logger_, "request_timeout is too small (< 100), use 100 instead.");
        request_timeout = 100;
    }
    // check
    bool succ_flag = true;
    if (http_host.empty()) {
        LOG4CXX_ERROR(logger_, "http_host is not set.");
        succ_flag = false;
    }
    if (http_port < 0) {
        LOG4CXX_ERROR(logger_, "invalid http_port: " << http_port);
        succ_flag = false;
    }
    return succ_flag;
}

void diffraflow::CtrConfig::print() {
    cout << "Configuration:" << endl;
    cout << "- http_host = " << http_host << endl;
    cout << "- http_port = " << http_port << endl;
    cout << "- request_timeout = " << request_timeout << endl;
}
