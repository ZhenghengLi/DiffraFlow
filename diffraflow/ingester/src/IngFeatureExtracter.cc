#include "IngFeatureExtracter.hh"
#include "IngImgFtrBuffer.hh"

log4cxx::LoggerPtr diffraflow::IngFeatureExtracter::logger_ = log4cxx::Logger::getLogger("IngFeatureExtracter");

diffraflow::IngFeatureExtracter::IngFeatureExtracter(
    IngImgFtrBuffer* buffer, IngBufferItemQueue* queue_in, IngBufferItemQueue* queue_out)
    : image_feature_buffer_(buffer), item_queue_in_(queue_in), item_queue_out_(queue_out) {
    worker_status_ = kNotStart;
}

diffraflow::IngFeatureExtracter::~IngFeatureExtracter() {}

void diffraflow::IngFeatureExtracter::extract_feature_(const shared_ptr<IngBufferItem>& item) {
    // some example code
    image_feature_buffer_->image_feature_host(item->index)->peak_counts = 1;
    image_feature_buffer_->image_feature_host(item->index)->global_rms = 2;
}

int diffraflow::IngFeatureExtracter::run_() {
    int result = 0;
    worker_status_ = kRunning;
    cv_status_.notify_all();
    shared_ptr<IngBufferItem> item;
    while (worker_status_ != kStopped && item_queue_in_->take(item)) {
        extract_feature_(item);
        if (item_queue_out_->push(item)) {
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
    worker_ = async(std::launch::async, &IngFeatureExtracter::run_, this);
    unique_lock<mutex> ulk(mtx_status_);
    cv_status_.wait(ulk, [this]() { return worker_status_ != kNotStart; });
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
