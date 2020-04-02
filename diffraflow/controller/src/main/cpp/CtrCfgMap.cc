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
    list< pair<string, string> > conf_KV_list;
    if (!read_conf_KV_list_(filename, conf_KV_list)) {
        LOG4CXX_ERROR(logger_, "Failed to read configuration file: " << filename);
        return false;
    }
    // parse
    data.clear();
    mtime = time(NULL);
    for (list< pair<string, string> >::iterator iter = conf_KV_list.begin(); iter != conf_KV_list.end(); ++iter) {
        string key = iter->first;
        string value = iter->second;
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