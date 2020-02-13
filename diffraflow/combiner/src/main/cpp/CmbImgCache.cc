#include "CmbImgCache.hh"
#include "ImageFrame.hh"
#include "ImageData.hh"

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

log4cxx::LoggerPtr diffraflow::CmbImgCache::logger_
    = log4cxx::Logger::getLogger("CmbImgCache");

diffraflow::CmbImgCache::CmbImgCache(size_t num_of_dets, size_t img_q_ms) {
    imgfrm_queues_len_ = num_of_dets;
    imgfrm_queues_arr_ = new queue<ImageFrame>[imgfrm_queues_len_];
    imgdat_queue.set_maxsize(img_q_ms);
}

diffraflow::CmbImgCache::~CmbImgCache() {

}

void diffraflow::CmbImgCache::push_frame(const ImageFrame& image_frame) {
    // in this function, put image from into priority queue, then try to do time alignment
    lock_guard<mutex> lk(mtx_);
    if (image_frame.det_id > imgfrm_queues_len_) {
        LOG4CXX_WARN(logger_, "Detector ID is out of range: " << image_frame.det_id);
        return;
    }
    imgfrm_queues_arr_[image_frame.det_id].push(image_frame);
    if (do_alignment_()) {
        // for debug only
        LOG4CXX_INFO(logger_, "Successfully do one alignment.");
    }
}

bool diffraflow::CmbImgCache::do_alignment_() {
    // in this function, try to do time alignment, and put the aligned full image data into imgdat_queue_
    if (imgfrm_queues_arr_[0].empty()) {
        return false;
    }
    ImageData image_data;
    image_data.put_imgfrm(0, imgfrm_queues_arr_[0].front());
    imgfrm_queues_arr_[0].pop();
    image_data.print();
    // imgdat_queue_.push(image_data);
    return true;
}
