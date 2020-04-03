#include "IngImgDatRawQueue.hh"

using std::lock_guard;

log4cxx::LoggerPtr diffraflow::IngImgDatRawQueue::logger_
    = log4cxx::Logger::getLogger("IngImgDatRawQueue");

diffraflow::IngImgDatRawQueue::IngImgDatRawQueue(size_t img_q_ms) {
    imgdat_queue_.set_maxsize(img_q_ms);
}

diffraflow::IngImgDatRawQueue::~IngImgDatRawQueue() {

}

bool diffraflow::IngImgDatRawQueue::push(const ImageData& image_data) {
    return imgdat_queue_.push(image_data);
}

bool diffraflow::IngImgDatRawQueue::take(ImageData& image_data) {
    bool result = imgdat_queue_.take(image_data);
    if (stopped_ && imgdat_queue_.empty()) {
        stop_cv_.notify_all();
    }
    return result;
}

void diffraflow::IngImgDatRawQueue::stop(int wait_time) {
    stopped_ = true;
    imgdat_queue_.stop();
    if (wait_time > 0) {
        unique_lock<mutex> ulk(stop_mtx_);
        stop_cv_.wait_for(ulk, std::chrono::milliseconds(wait_time),
            [this]() {return imgdat_queue_.empty();}
        );
    }
}
