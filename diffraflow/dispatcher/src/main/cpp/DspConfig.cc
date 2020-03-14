#include "DspConfig.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <boost/algorithm/string.hpp>

using std::cout;
using std::flush;
using std::endl;

log4cxx::LoggerPtr diffraflow::DspConfig::logger_
    = log4cxx::Logger::getLogger("DspConfig");

diffraflow::DspConfig::DspConfig() {
    dispatcher_id = 0;
    listen_host = "0.0.0.0";
    listen_port = -1;
    compress_method = DspSender::kNone;
}

diffraflow::DspConfig::~DspConfig() {

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
        } else if (key == "compress_method") {
            if (value == "LZ4") {
                compress_method = DspSender::kLZ4;
            } else if (value == "Snappy") {
                compress_method = DspSender::kSnappy;
            } else if (value == "ZSTD") {
                compress_method = DspSender::kZSTD;
            } else {
                compress_method = DspSender::kNone;
            }
        } else if (key == "compress_level") {
            compress_level = atoi(value.c_str());
        } else {
            LOG4CXX_WARN(logger_, "Found unknown configuration which is ignored: "
                << key << " = " << value << " in " << filename);
        }
    }
    // use POD IP as dispatcher_id
    if (dispatcher_id == 0) {
        const char* pod_ip = getenv("POD_IP");
        if (pod_ip != NULL) {
            vector<string> ip_nums;
            boost::split(ip_nums, pod_ip, boost::is_any_of("."));
            for (size_t i = 0; i < ip_nums.size(); i++) {
                dispatcher_id <<= 8;
                dispatcher_id += atoi(ip_nums[i].c_str());
            }
        }
    }
    // check
    bool succ_flag = true;
    if (listen_port < 0) {
        LOG4CXX_ERROR(logger_, "invalid listen_port: " << listen_port);
        succ_flag = false;
    }
    if (compress_method == DspSender::kZSTD && (compress_level < 1 || compress_level >= 20)) {
        LOG4CXX_ERROR(logger_, "compress level for ZSTD compress method is out of range, it should be >= 1 and < 20.");
        succ_flag = false;
    }
    return succ_flag;
}

void diffraflow::DspConfig::print() {
    cout << " ---- Configuration Dump Begin ----" << endl;
    cout << "  listen_port = " << listen_port << endl;
    cout << "  dispatcher_id = " << dispatcher_id << endl;
    cout << "  compress_method = " << flush;
    switch (compress_method) {
    case DspSender::kLZ4:
        cout << "LZ4" << endl;
        break;
    case DspSender::kSnappy:
        cout << "Snappy" << endl;
        break;
    case DspSender::kZSTD:
        cout << "ZSTD (" << compress_level << ")" << endl;
        break;
    default:
        cout << "None" << endl;
    }
    cout << " ---- Configuration Dump End ----" << endl;
}