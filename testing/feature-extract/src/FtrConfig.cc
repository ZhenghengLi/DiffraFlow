#include "FtrConfig.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <regex>
#include <boost/algorithm/string.hpp>
#include <thread>

using std::cout;
using std::flush;
using std::endl;
using std::regex;
using std::regex_match;
using std::regex_replace;
using std::ifstream;

log4cxx::LoggerPtr diffraflow::FtrConfig::logger_ = log4cxx::Logger::getLogger("FtrConfig");

diffraflow::FtrConfig::FtrConfig() {
    peak_min_energy = -100;
    peak_max_energy = 10000;
    peak_inlier_thr = 2;
    peak_outlier_thr = 10;
    peak_residual_thr = 500;
    peak_energy_thr = 0;
}

diffraflow::FtrConfig::~FtrConfig() {}

bool diffraflow::FtrConfig::load(const char* filename) {
    list<pair<string, string>> conf_KV_list;
    if (!read_conf_KV_list_(filename, conf_KV_list)) {
        LOG4CXX_ERROR(logger_, "Failed to read configuration file: " << filename);
        return false;
    }
    // parse
    for (list<pair<string, string>>::iterator iter = conf_KV_list.begin(); iter != conf_KV_list.end(); ++iter) {
        string key = iter->first;
        string value = iter->second;
        if (key == "peak_min_energy") {
            peak_min_energy = atof(value.c_str());
        } else if (key == "peak_inlier_thr") {
            peak_inlier_thr = atof(value.c_str());
        } else if (key == "peak_outlier_thr") {
            peak_outlier_thr = atoi(value.c_str());
        } else if (key == "peak_residual_thr") {
            peak_residual_thr = atoi(value.c_str());
        } else if (key == "peak_energy_thr") {
            peak_energy_thr = atoi(value.c_str());
        } else {
            LOG4CXX_WARN(logger_,
                "Found unknown configuration which is ignored: " << key << " = " << value << " in " << filename);
        }
    }

    // correction

    // check
    bool succ_flag = true;

    return succ_flag;
}

void diffraflow::FtrConfig::print() {
    cout << " ---- Configuration Dump Begin ----" << endl;
    cout << " peak_min_energy    = " << peak_min_energy << endl;
    cout << " peak_max_energy    = " << peak_max_energy << endl;
    cout << " peak_inlier_thr    = " << peak_inlier_thr << endl;
    cout << " peak_outlier_thr   = " << peak_outlier_thr << endl;
    cout << " peak_residual_thr  = " << peak_residual_thr << endl;
    cout << " peak_energy_thr    = " << peak_energy_thr << endl;
    cout << " ---- Configuration Dump End ----" << endl;
}
