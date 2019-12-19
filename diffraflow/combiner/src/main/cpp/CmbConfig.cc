#include "CmbConfig.hh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/log/trivial.hpp>

using std::ifstream;
using std::stringstream;
using std::cout;
using std::cerr;
using std::endl;

diffraflow::CmbConfig::CmbConfig() {

}

diffraflow::CmbConfig::~CmbConfig() {

}

bool diffraflow::CmbConfig::load(const char* filename) {
    vector< pair<string, string> > conf_KV_vec;
    if (!read_conf_KV_vec_(filename, conf_KV_vec)) {
        BOOST_LOG_TRIVIAL(error) << "Failed to read configuration file: " << filename;
        return false;
    }
    for (size_t i = 0; i < conf_KV_vec.size(); i++) {
        string key = conf_KV_vec[i].first;
        string value = conf_KV_vec[i].second;
        if (key == "port") {
            port = atoi(value.c_str());
        }
    }
    return true;
}

void diffraflow::CmbConfig::print() {
    cout << " = CmbConfig Dump Begin =" << endl;
    cout << "  port = " << port << endl;
    cout << " = CmbConfig Dump End =" << endl;

}
