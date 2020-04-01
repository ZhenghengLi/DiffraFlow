#include "CmbSrvMan.hh"
#include "CmbConfig.hh"
#include "CmbImgCache.hh"
#include "CmbImgFrmSrv.hh"
#include "CmbImgDatSrv.hh"

log4cxx::LoggerPtr diffraflow::CmbSrvMan::logger_
    = log4cxx::Logger::getLogger("CmbSrvMan");

diffraflow::CmbSrvMan::CmbSrvMan(CmbConfig* config) {
    config_obj_ = config;
    running_flag_ = false;
    image_cache_ = nullptr;
    imgfrm_srv_ = nullptr;
    imgdat_srv_ = nullptr;
}

diffraflow::CmbSrvMan::~CmbSrvMan() {

}

void diffraflow::CmbSrvMan::start_run() {
    if (running_flag_) return;

    image_cache_ = new CmbImgCache(1);

    imgfrm_srv_ = new CmbImgFrmSrv(config_obj_->imgfrm_listen_host,
        config_obj_->imgfrm_listen_port, image_cache_);
    imgdat_srv_ = new CmbImgDatSrv(config_obj_->imgdat_listen_host,
        config_obj_->imgdat_listen_port, image_cache_);

    // multiple servers start from here
    if (imgfrm_srv_->start()) {
        LOG4CXX_INFO(logger_, "successfully started image frame server.")
    } else {
        LOG4CXX_ERROR(logger_, "failed to start image frame server.")
        return;
    }
    if (imgdat_srv_->start()) {
        LOG4CXX_INFO(logger_, "successfully started image data server.")
    } else {
        LOG4CXX_ERROR(logger_, "failed to start image data server.")
        return;
    }

    running_flag_ = true;

    // then wait for finishing
    async([this]() {
        imgfrm_srv_->wait();
        imgdat_srv_->wait();
    }).wait();

}

void diffraflow::CmbSrvMan::terminate() {
    if (!running_flag_) return;

    // stop image frame server
    int result = imgfrm_srv_->stop_and_close();
    if (result == 0) {
        LOG4CXX_INFO(logger_, "image frame server is normally terminated.");
    } else if (result > 0) {
        LOG4CXX_WARN(logger_, "image frame server is abnormally terminated with error code: " << result);
    } else {
        LOG4CXX_WARN(logger_, "image frame server has not yet been started or already been closed.");
    }
    // delete image frame server
    delete imgfrm_srv_;
    imgfrm_srv_ = nullptr;

    // stop image cache
    image_cache_->stop(/* wait time, default is 0 */);

    // stop image data server
    result = imgdat_srv_->stop_and_close();
    if (result == 0) {
        LOG4CXX_INFO(logger_, "image data server is normally terminated.");
    } else if (result > 0) {
        LOG4CXX_WARN(logger_, "image data server is abnormally terminated with error code: " << result);
    } else {
        LOG4CXX_WARN(logger_, "image data server has not yet been started or already been closed.");
    }
    // delete image data server
    delete imgdat_srv_;
    imgdat_srv_ = nullptr;

    // delete image cache;
    delete image_cache_;
    image_cache_ = nullptr;

    running_flag_ = false;
}
