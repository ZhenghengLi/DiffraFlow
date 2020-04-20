#include "MonSrvMan.hh"
#include "MonConfig.hh"

log4cxx::LoggerPtr diffraflow::MonSrvMan::logger_
    = log4cxx::Logger::getLogger("MonSrvMan");

diffraflow::MonSrvMan::MonSrvMan(MonConfig* config) {
    config_obj_ = config;
    running_flag_ = false;
}

diffraflow::MonSrvMan::~MonSrvMan() {

}

void diffraflow::MonSrvMan::start_run() {
    if (running_flag_) return;


    running_flag_ = true;

    // then wait for finishing
    // async([this]() {

    // }).wait();

}

void diffraflow::MonSrvMan::terminate() {
    if (!running_flag_) return;


    running_flag_ = false;
}
