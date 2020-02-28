#include "IngConfig.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <iostream>

using std::cout;
using std::endl;

log4cxx::LoggerPtr diffraflow::IngConfig::logger_
    = log4cxx::Logger::getLogger("IngConfig");

diffraflow::IngConfig::IngConfig() {

}

diffraflow::IngConfig::~IngConfig() {

}

bool diffraflow::IngConfig::load(const char* filename) {
    if (!DynamicConfiguration::load(filename)) {
        return true;
    }
    for (map<string, string>::iterator iter = conf_map_.begin(); iter != conf_map_.end(); ++iter) {

    }
    return true;
}

void diffraflow::IngConfig::print() {

}

void diffraflow::IngConfig::convert_and_check() {

}