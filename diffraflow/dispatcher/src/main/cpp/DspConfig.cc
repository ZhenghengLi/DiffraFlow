#include "DspConfig.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <iostream>
#include <sstream>

using std::cout;
using std::endl;

diffraflow::DspConfig::DspConfig() {
    dispatcher_id = -1;
    listen_port = -1;
    combiner_address_file = "";
    compress_flag = false;
    logger_ = log4cxx::Logger::getLogger("DspConfig");
}

diffraflow::DspConfig::~DspConfig() {
    log4cxx::NDC::remove();
}

bool diffraflow::DspConfig::load(const char* filename) {
    vector< pair<string, string> > conf_KV_vec;
    if (!read_conf_KV_vec_(filename, conf_KV_vec)) {
        LOG4CXX_ERROR(logger_, "Failed to read configuration file: " << filename);
        return false;
    }
    // parse
    for (size_t i = 0; i < conf_KV_vec.size(); i++) {
        string key = conf_KV_vec[i].first;
        string value = conf_KV_vec[i].second;
        if (key == "listen_host") {
            listen_host = value;
        } else if (key == "listen_port") {
            listen_port = atoi(value.c_str());
        } else if (key == "dispatcher_id") {
            dispatcher_id = atoi(value.c_str());
        } else if (key == "combiner_address_file") {
            combiner_address_file = value;
        } else if (key == "compress_flag") {
            std::istringstream(value) >> std::boolalpha >> compress_flag;
        } else {
            LOG4CXX_WARN(logger_, "Found unknown configuration which is ignored: "
                << key << " = " << value << " in " << filename);
        }
    }
    // check
    bool succ_flag = true;
    if (dispatcher_id < 0) {
        LOG4CXX_ERROR(logger_, "invalid dispatcher_id: " << dispatcher_id);
        succ_flag = false;
    }
    if (listen_port < 0) {
        LOG4CXX_ERROR(logger_, "invalid listen_port: " << listen_port);
        succ_flag = false;
    }
    if (combiner_address_file == "") {
        LOG4CXX_ERROR(logger_, "combiner_address_file is not set");
        succ_flag = false;
    }
    return succ_flag;
}

void diffraflow::DspConfig::print() {
    cout << " ---- Configuration Dump Begin ----" << endl;
    cout << "  listen_port = " << listen_port << endl;
    cout << "  dispatcher_id = " << dispatcher_id << endl;
    cout << "  combiner_address_file = " << combiner_address_file << endl;
    cout << " ---- Configuration Dump End ----" << endl;
}