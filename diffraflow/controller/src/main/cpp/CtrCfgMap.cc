#include "CtrCfgMap.hh"
#include <iostream>

using std::cout;
using std::endl;
using std::flush;

log4cxx::LoggerPtr diffraflow::CtrCfgMap::logger_
    = log4cxx::Logger::getLogger("CtrCfgMap");

diffraflow::CtrCfgMap::CtrCfgMap() {

}

diffraflow::CtrCfgMap::~CtrCfgMap() {

}

bool diffraflow::CtrCfgMap::load(const char* filename) {
    vector< pair<string, string> > conf_KV_vec;
    if (!read_conf_KV_vec_(filename, conf_KV_vec)) {
        LOG4CXX_ERROR(logger_, "Failed to read configuration file: " << filename);
        return false;
    }
    // parse
    data.clear();
    mtime = time(NULL);
    for (size_t i = 0; i < conf_KV_vec.size(); i++) {
        string key = conf_KV_vec[i].first;
        string value = conf_KV_vec[i].second;
        data[key] = value;
    }
    return true;
}

void diffraflow::CtrCfgMap::print() {
    if (data.empty()) {
        cout << "config_map has no data." << endl;
        return;
    }
    cout << "config_map:" << endl;
    for (map<string, string>::iterator iter = data.begin(); iter != data.end(); ++iter) {
        cout << "- " << iter->first << " = " << iter->second << endl;
    }
    cout << "config_mtime: " << ctime(&mtime) << flush;
}