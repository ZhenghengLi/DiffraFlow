#include "CmbSrvMan.hh"
#include "CmbConfig.hh"
#include "CmbImgCache.hh"
#include "CmbImgFrmSrv.hh"

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
    imgfrm_srv_ = new CmbImgFrmSrv(config_obj_->port, image_cache_);
    running_flag_ = true;
    imgfrm_srv_->serve();
}

void diffraflow::CmbSrvMan::terminate() {
    if (!running_flag_) return;
    imgfrm_srv_->stop();
    delete imgfrm_srv_;
    imgfrm_srv_ = nullptr;
    delete image_cache_;
    image_cache_ = nullptr;
    running_flag_ = false;
}
