#include "DspConfig.hh"
#include <boost/log/trivial.hpp>

diffraflow::DspConfig::DspConfig() {
    dispatcher_id = 1234;
    listen_port = 2727;
}

diffraflow::DspConfig::~DspConfig() {

}

bool diffraflow::DspConfig::load(const char* filename) {
    vector< pair<string, string> > conf_KV_vec;
    if (!read_conf_KV_vec_(filename, conf_KV_vec)) {
        BOOST_LOG_TRIVIAL(error) << "Failed to read configuration file: " << filename;
        return false;
    }
    for (size_t i = 0; i < conf_KV_vec.size(); i++) {
        string key = conf_KV_vec[i].first;
        string value = conf_KV_vec[i].second;
        if (key == "listen_port") {
            listen_port = atoi(value.c_str());
        } else if (key == "dispatcher_id") {
            dispatcher_id = atoi(value.c_str());
        }
    }
    return true;
    return true;
}

void diffraflow::DspConfig::print() {

}