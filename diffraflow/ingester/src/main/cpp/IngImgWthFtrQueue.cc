#include "IngImgWthFtrQueue.hh"

log4cxx::LoggerPtr diffraflow::IngImgWthFtrQueue::logger_ = log4cxx::Logger::getLogger("IngImgWthFtrQueue");

diffraflow::IngImgWthFtrQueue::IngImgWthFtrQueue(size_t img_q_ms) { imgWthFtr_queue_.set_maxsize(img_q_ms); }

diffraflow::IngImgWthFtrQueue::~IngImgWthFtrQueue() {}

bool diffraflow::IngImgWthFtrQueue::push(const shared_ptr<ImageWithFeature>& image_with_feature) {
    return imgWthFtr_queue_.push(image_with_feature);
}

bool diffraflow::IngImgWthFtrQueue::take(shared_ptr<ImageWithFeature>& image_with_feature) {
    bool result = imgWthFtr_queue_.take(image_with_feature);
    if (stopped_ && imgWthFtr_queue_.empty()) {
        stop_cv_.notify_all();
    }
    return result;
}

void diffraflow::IngImgWthFtrQueue::stop(int wait_time) {
    stopped_ = true;
    imgWthFtr_queue_.stop();
    if (wait_time > 0) {
        unique_lock<mutex> ulk(stop_mtx_);
        stop_cv_.wait_for(ulk, std::chrono::milliseconds(wait_time), [this]() { return imgWthFtr_queue_.empty(); });
    }
}
