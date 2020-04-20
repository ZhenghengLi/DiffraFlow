#include "IngImageFilter.hh"
#include "IngImgWthFtrQueue.hh"
#include "IngConfig.hh"

log4cxx::LoggerPtr diffraflow::IngImageFilter::logger_
    = log4cxx::Logger::getLogger("IngImageFilter");

diffraflow::IngImageFilter::IngImageFilter(
    IngImgWthFtrQueue* img_queue_in, IngImgWthFtrQueue* img_queue_out, IngConfig* conf_obj) {
    image_queue_in_ = img_queue_in;
    image_queue_out_ = img_queue_out;
    config_obj_ = conf_obj;
    worker_status_ = kNotStart;
}

diffraflow::IngImageFilter::~IngImageFilter() {

}

bool diffraflow::IngImageFilter::check_for_save_(const ImageFeature& image_feature) {
    return true;
}

bool diffraflow::IngImageFilter::check_for_monitor_(const ImageFeature& image_feature) {
    return true;
}

int diffraflow::IngImageFilter::run_() {
    int result = 0;
    worker_status_ = kRunning;
    cv_status_.notify_all();
    shared_ptr<ImageWithFeature> image_with_feature;
    while (worker_status_ != kStopped && image_queue_in_->take(image_with_feature)) {
        if (check_for_save_(image_with_feature->image_feature)) {
            if (image_queue_out_->push(image_with_feature)) {
                LOG4CXX_DEBUG(logger_, "pushed the one good image into queue for saving.");
            } else {
                break;
            }
        }
        if (check_for_monitor_(image_with_feature->image_feature)) {
            set_current_image(image_with_feature);
        }
    }
    worker_status_ = kStopped;
    cv_status_.notify_all();
    return result;
}

bool diffraflow::IngImageFilter::start() {
    if (!(worker_status_ == kNotStart || worker_status_ == kStopped)) {
        return false;
    }
    worker_status_ = kNotStart;
    worker_ = async(&IngImageFilter::run_, this);
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

void diffraflow::IngImageFilter::wait() {
    if (worker_.valid()) {
        worker_.wait();
    }
}

int diffraflow::IngImageFilter::stop() {
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

void diffraflow::IngImageFilter::set_current_image(const shared_ptr<ImageWithFeature>& image_with_feature) {
    unique_lock<mutex> lk(current_image_mtx_, std::try_to_lock);
    if (lk.owns_lock()) {
        current_image_ = image_with_feature;
    }
}

bool diffraflow::IngImageFilter::get_current_image(ImageWithFeature& image_with_feature) {
    lock_guard<mutex> lg(current_image_mtx_);
    if (current_image_) {
        image_with_feature = *current_image_;
        return true;
    } else {
        return false;
    }
}
