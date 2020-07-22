#include "SndSrvMan.hh"
#include "SndConfig.hh"
#include "SndDatTran.hh"
#include "SndTrgSrv.hh"

#include <fstream>
#include <string>

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <boost/algorithm/string.hpp>

using std::ifstream;
using std::make_pair;

log4cxx::LoggerPtr diffraflow::SndSrvMan::logger_ = log4cxx::Logger::getLogger("SndSrvMan");

diffraflow::SndSrvMan::SndSrvMan(SndConfig* config) {
    config_obj_ = config;
    data_transfer_ = nullptr;
    trigger_srv_ = nullptr;
    running_flag_ = false;
}

diffraflow::SndSrvMan::~SndSrvMan() {}

void diffraflow::SndSrvMan::start_run() {
    if (running_flag_) return;
    // create data transfer
    data_transfer_ = new SndDatTran(config_obj_);
    if (data_transfer_->connect_to_server()) {
        LOG4CXX_INFO(logger_, "successfully connected to dispatcher " << config_obj_->dispatcher_host << ":"
                                                                      << config_obj_->dispatcher_port << ".");
    } else {
        LOG4CXX_ERROR(logger_, "failed to connect to dispatcher " << config_obj_->dispatcher_host << ":"
                                                                  << config_obj_->dispatcher_port << ".");
        return;
    }
    trigger_srv_ = new SndTrgSrv(config_obj_->dispatcher_host, config_obj_->dispatcher_port, data_transfer_);

    // multiple servers start from here
    if (trigger_srv_->start()) {
        LOG4CXX_INFO(logger_, "successfully started trigger server.")
    } else {
        LOG4CXX_ERROR(logger_, "failed to start trigger server.")
        return;
    }

    running_flag_ = true;

    // wait for finishing
    async([this]() { trigger_srv_->wait(); }).wait();
}

void diffraflow::SndSrvMan::terminate() {
    if (!running_flag_) return;

    // stop and delete trigger server
    int result = trigger_srv_->stop_and_close();
    if (result == 0) {
        LOG4CXX_INFO(logger_, "trigger server is normally terminated.");
    } else if (result > 0) {
        LOG4CXX_WARN(logger_, "trigger server is abnormally terminated with error code: " << result);
    } else {
        LOG4CXX_WARN(logger_, "trigger server has not yet been started or already been closed.");
    }
    delete trigger_srv_;
    trigger_srv_ = nullptr;

    running_flag_ = false;
}
