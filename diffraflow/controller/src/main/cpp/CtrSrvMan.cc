#include "CtrSrvMan.hh"
#include "CtrConfig.hh"

#include <future>

using std::async;

log4cxx::LoggerPtr diffraflow::CtrSrvMan::logger_
    = log4cxx::Logger::getLogger("CtrSrvMan");

diffraflow::CtrSrvMan::CtrSrvMan(CtrConfig* config, const char* monaddr_file) {
    config_obj_ = config;
    monitor_address_file_ = monaddr_file;
    running_flag_ = false;
}

diffraflow::CtrSrvMan::~CtrSrvMan() {

}

void diffraflow::CtrSrvMan::start_run() {
    if (running_flag_) return;



    running_flag_ = true;

    // then wait for finishing
    // async([this]() {

    // }).wait();

}

void diffraflow::CtrSrvMan::terminate() {
    if (!running_flag_) return;


    running_flag_ = false;
}
