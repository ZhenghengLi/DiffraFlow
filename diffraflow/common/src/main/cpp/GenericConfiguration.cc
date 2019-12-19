#include "GenericConfiguration.hh"
#include <iostream>
#include <fstream>

#include <boost/log/trivial.hpp>
#include <boost/algorithm/string.hpp>

using std::ifstream;

diffraflow::GenericConfiguration::GenericConfiguration() {

}

diffraflow::GenericConfiguration::~GenericConfiguration() {

}

bool diffraflow::GenericConfiguration::read_conf_KV_vec_(
    const char* filename, vector< pair<string, string> >& conf_KV_vec) {
    conf_KV_vec.clear();
    ifstream conf_file;
    conf_file.open(filename);
    if (!conf_file.is_open()) {
        BOOST_LOG_TRIVIAL(error) << "configuration file open failed.";
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
        // extract host and port
        vector<string> key_value;
        boost::split(key_value, oneline, boost::is_any_of("="));
        if (key_value.size() < 2) {
            BOOST_LOG_TRIVIAL(error) << "found unknown configuration: " << oneline;
            return false;
        }
        boost::trim(key_value[0]);
        boost::trim(key_value[1]);
        if (key_value[0].size() > 0 && key_value[1].size() > 0) {
            conf_KV_vec.push_back(make_pair(key_value[0], key_value[1]));
        }
    }
    if (conf_KV_vec.size() > 0) {
        return true;
    } else {
        BOOST_LOG_TRIVIAL(warning) << "there is no valid configurations in file: " << filename;
        return false;
    }
}
