#include "CmbSrvMan.hh"
#include "CmbConfig.hh"
#include "CmbImgCache.hh"
#include "CmbImgFrmSrv.hh"

log4cxx::LoggerPtr diffraflow::CmbSrvMan::logger_
    = log4cxx::Logger::getLogger("CmbSrvMan");

diffraflow::CmbSrvMan::CmbSrvMan(CmbConfig* config) {
    config_obj_ = config;
    running_flag_ = false;
    image_cache_ = nullptr;
    imgfrm_srv_ = nullptr;
}

diffraflow::CmbSrvMan::~CmbSrvMan() {

}

void diffraflow::CmbSrvMan::start_run() {
    if (running_flag_) return;
    image_cache_ = new CmbImgCache(1);
    imgfrm_srv_ = new CmbImgFrmSrv(config_obj_->listen_host,
        config_obj_->listen_port, image_cache_);
    running_flag_ = true;

    // multiple servers start from here
    imgfrm_srv_->start();

    // then wait for finishing
    async([this]() {
        imgfrm_srv_->wait();
    }).wait();

}

void diffraflow::CmbSrvMan::terminate() {
    if (!running_flag_) return;

    imgfrm_srv_->stop();
    int result = imgfrm_srv_->get();
    if (result == 0) {
        LOG4CXX_INFO(logger_, "image frame server is normally terminated.");
    } else if (result > 0) {
        LOG4CXX_WARN(logger_, "image frame server is abnormally terminated with error code: " << result);
    } else {
        LOG4CXX_WARN(logger_, "image frame server has not yet been started.");
    }
    delete imgfrm_srv_;
    imgfrm_srv_ = nullptr;

    delete image_cache_;
    image_cache_ = nullptr;
    running_flag_ = false;
}
