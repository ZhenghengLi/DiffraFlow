#include "IngImgDatRawQueue.hh"

using std::lock_guard;

log4cxx::LoggerPtr diffraflow::IngImgDatRawQueue::logger_
    = log4cxx::Logger::getLogger("IngImgDatRawQueue");

diffraflow::IngImgDatRawQueue::IngImgDatRawQueue() {

}

diffraflow::IngImgDatRawQueue::~IngImgDatRawQueue() {

}

bool diffraflow::IngImgDatRawQueue::push(const ImageData& image_data) {
    if (image_data.late_arrived) {
        return imgdat_queue_late_.push(image_data);
    } else {
        return imgdat_queue_.push(image_data);
    }
}

bool diffraflow::IngImgDatRawQueue::take(ImageData& image_data) {
    bool result = false;
    if (imgdat_queue_late_.empty()) {
        result = imgdat_queue_.take(image_data);
    } else {
        result = imgdat_queue_late_.take(image_data);
    }
    if (stopped_ && imgdat_queue_late_.empty() && imgdat_queue_.empty()) {
        stop_cv_.notify_all();
    }
    return result;
}

void diffraflow::IngImgDatRawQueue::stop(int wait_time) {
    stopped_ = true;

    imgdat_queue_late_.stop();
    imgdat_queue_.stop();

    if (wait_time > 0) {
        unique_lock<mutex> ulk(stop_mtx_);
        stop_cv_.wait_for(ulk, std::chrono::milliseconds(wait_time),
            [this]() {return imgdat_queue_late_.empty() && imgdat_queue_.empty();}
        );
    }
}
