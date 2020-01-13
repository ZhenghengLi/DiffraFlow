#include "DspConfig.hh"
#include <boost/log/trivial.hpp>
#include <iostream>
#include <sstream>

using std::cout;
using std::endl;

diffraflow::DspConfig::DspConfig() {
    dispatcher_id = -1;
    listen_port = -1;
    combiner_address_file = "";
    compress_flag = false;
}

diffraflow::DspConfig::~DspConfig() {

}

bool diffraflow::DspConfig::load(const char* filename) {
    vector< pair<string, string> > conf_KV_vec;
    if (!read_conf_KV_vec_(filename, conf_KV_vec)) {
        BOOST_LOG_TRIVIAL(error) << "Failed to read configuration file: " << filename;
        return false;
    }
    // parse
    for (size_t i = 0; i < conf_KV_vec.size(); i++) {
        string key = conf_KV_vec[i].first;
        string value = conf_KV_vec[i].second;
        if (key == "listen_port") {
            listen_port = atoi(value.c_str());
        } else if (key == "dispatcher_id") {
            dispatcher_id = atoi(value.c_str());
        } else if (key == "combiner_address_file") {
            combiner_address_file = value;
        } else if (key == "compress_flag") {
            std::istringstream(value) >> std::boolalpha >> compress_flag;
        } else {
            BOOST_LOG_TRIVIAL(warning) << "Found unknown configuration which is ignored: "
                << key << " = " << value << " in " << filename;
        }
    }
    // check
    bool succ_flag = true;
    if (dispatcher_id < 0) {
        BOOST_LOG_TRIVIAL(error) << "invalid dispatcher_id: " << dispatcher_id;
        succ_flag = false;
    }
    if (listen_port < 0) {
        BOOST_LOG_TRIVIAL(error) << "invalid listen_port: " << listen_port;
        succ_flag = false;
    }
    if (combiner_address_file == "") {
        BOOST_LOG_TRIVIAL(error) << "combiner_address_file is not set";
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