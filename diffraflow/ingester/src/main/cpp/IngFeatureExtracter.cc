#include "IngFeatureExtracter.hh"
#include "IngImgWthFtrQueue.hh"

log4cxx::LoggerPtr diffraflow::IngFeatureExtracter::logger_
    = log4cxx::Logger::getLogger("IngFeatureExtracter");

diffraflow::IngFeatureExtracter::IngFeatureExtracter(
    IngImgWthFtrQueue* img_queue_in, IngImgWthFtrQueue* img_queue_out) {
    image_queue_in_ = img_queue_in;
    image_queue_out_ = img_queue_out;
    worker_status_ = kNotStart;
}

diffraflow::IngFeatureExtracter::~IngFeatureExtracter() {

}

void diffraflow::IngFeatureExtracter::extract_feature_(
    const ImageData& imgdat_raw, ImageFeature& image_feature) {

    // some example code
    image_feature.peak_counts = 1;
    image_feature.global_rms = 2;
    image_feature.set_defined();

}

int diffraflow::IngFeatureExtracter::run_() {
    int result = 0;
    worker_status_ = kRunning;
    cv_status_.notify_all();
    shared_ptr<ImageWithFeature> image_with_feature;
    while (worker_status_ != kStopped && image_queue_in_->take(image_with_feature)) {
        extract_feature_(image_with_feature->image_data_calib, image_with_feature->image_feature);
        if (image_queue_out_->push(image_with_feature)) {
            LOG4CXX_DEBUG(logger_, "pushed the feature data into queue.");
        } else {
            break;
        }
    }
    worker_status_ = kStopped;
    cv_status_.notify_all();
    return result;
}

bool diffraflow::IngFeatureExtracter::start() {
    if (!(worker_status_ == kNotStart || worker_status_ == kStopped)) {
        return false;
    }
    worker_status_ = kNotStart;
    worker_ = async(&IngFeatureExtracter::run_, this);
    unique_lock<mutex> ulk(mtx_status_);
    cv_status_.wait(ulk,
        [this]() {
            return worker_status_ != kNotStart;
        }
    );
    if (worker_status_ == kRunning) {
        return true;
    } else {
        return false;
    }

}

void diffraflow::IngFeatureExtracter::wait() {
    if (worker_.valid()) {
        worker_.wait();
    }
}

int diffraflow::IngFeatureExtracter::stop() {
    if (worker_status_ == kNotStart) {
        return -1;
    }
    worker_status_ = kStopped;
    cv_status_.notify_all();
    int result = -2;
    if (worker_.valid()) {
        result = worker_.get();
    }
    return result;
}
