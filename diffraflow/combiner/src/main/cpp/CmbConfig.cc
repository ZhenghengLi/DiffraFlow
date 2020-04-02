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
    imgfrm_listen_host = "0.0.0.0";
    imgfrm_listen_port = -1;
    imgdat_listen_host = "0.0.0.0";
    imgdat_listen_port = -1;
    imgdat_queue_capacity = 100;
}

diffraflow::CmbConfig::~CmbConfig() {

}

bool diffraflow::CmbConfig::load(const char* filename) {
    list< pair<string, string> > conf_KV_list;
    if (!read_conf_KV_list_(filename, conf_KV_list)) {
        LOG4CXX_ERROR(logger_, "Failed to read configuration file: " << filename);
        return false;
    }
    for (list< pair<string, string> >::iterator iter = conf_KV_list.begin(); iter != conf_KV_list.end(); ++iter) {
        string key = iter->first;
        string value = iter->second;
        if (key == "imgfrm_listen_host") {
            imgfrm_listen_host = value;
        } else if (key == "imgfrm_listen_port") {
            imgfrm_listen_port = atoi(value.c_str());
        } else if (key == "imgdat_listen_host") {
            imgdat_listen_host = value;
        } else if (key == "imgdat_listen_port") {
            imgdat_listen_port = atoi(value.c_str());
        } else if (key == "imgdat_queue_capacity") {
            imgdat_queue_capacity = atoi(value.c_str());
        } else {
            LOG4CXX_WARN(logger_, "Found unknown configuration which is ignored: "
                << key << " = " << value << " in " << filename);
        }
    }
    // check
    bool succ_flag = true;
    if (imgfrm_listen_port < 0) {
        LOG4CXX_ERROR(logger_, "invalid imgfrm_listen_port: " << imgfrm_listen_port);
        succ_flag = false;
    }
    if (imgdat_listen_port < 0) {
        LOG4CXX_ERROR(logger_, "invalid imgdat_listen_port: " << imgdat_listen_port);
        succ_flag = false;
    }
    if (imgdat_queue_capacity < 1 || imgdat_queue_capacity > 10000) {
        LOG4CXX_ERROR(logger_, "imgdat_queue_capacity is out of range " << 1 << "-" << 10000);
        succ_flag = false;
    }
    return succ_flag;
}

void diffraflow::CmbConfig::print() {
    cout << " = Configuration Dump Begin =" << endl;
    cout << "  imgfrm_listen_host = " << imgfrm_listen_host << endl;
    cout << "  imgfrm_listen_port = " << imgfrm_listen_port << endl;
    cout << "  imgdat_listen_host = " << imgdat_listen_host << endl;
    cout << "  imgdat_listen_port = " << imgdat_listen_port << endl;
    cout << "  imgdat_queue_capacity = " << imgdat_queue_capacity << endl;
    cout << " = Configuration Dump End =" << endl;

}
