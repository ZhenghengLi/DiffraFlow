#include "CmbImgCache.hh"
#include "ImageFrame.hh"
#include "ImageData.hh"

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <limits>

using std::numeric_limits;
using std::lock_guard;
using std::unique_lock;

log4cxx::LoggerPtr diffraflow::CmbImgCache::logger_
    = log4cxx::Logger::getLogger("CmbImgCache");

diffraflow::CmbImgCache::CmbImgCache(size_t num_of_dets, size_t img_q_ms) {
    imgfrm_queues_len_ = num_of_dets;
    imgfrm_queues_arr_ = new TimeOrderedQueue<ImageFrame, uint64_t>[imgfrm_queues_len_];
    imgdat_queue_.set_maxsize(img_q_ms);
    // wait forever
    wait_threshold_ = numeric_limits<uint64_t>::max();
    image_time_min_ = numeric_limits<uint64_t>::max();
    image_time_last_ = 0;
    num_of_empty_ = imgfrm_queues_len_;
    distance_max_ = 0;
    stopped_ = false;
}

diffraflow::CmbImgCache::~CmbImgCache() {

}

void diffraflow::CmbImgCache::push_frame(const ImageFrame& image_frame) {
    if (stopped_) return;
    // in this function, put image from into priority queue, then try to do time alignment
    lock_guard<mutex> lk(data_mtx_);

    if (image_frame.detector_id >= imgfrm_queues_len_) {
        LOG4CXX_WARN(logger_, "Detector ID is out of range: " << image_frame.detector_id);
        return;
    }

    if (image_frame.image_time < image_time_min_) {
        image_time_min_ = image_frame.image_time;
    }
    if (imgfrm_queues_arr_[image_frame.detector_id].empty()) {
        num_of_empty_--;
    }
    imgfrm_queues_arr_[image_frame.detector_id].push(image_frame);
    uint64_t distance_current = imgfrm_queues_arr_[image_frame.detector_id].distance();
    if (distance_current > distance_max_) {
        distance_max_ = distance_current;
    }

    while (do_alignment_(false));

}

bool diffraflow::CmbImgCache::do_alignment_(bool force_flag) {
    if (num_of_empty_ == imgfrm_queues_len_) {
        return false;
    }
    if (num_of_empty_ <= 0 || distance_max_ > wait_threshold_ || force_flag) {
        uint64_t image_time_target = image_time_min_;
        image_time_min_ = numeric_limits<uint64_t>::max();
        num_of_empty_ = 0;
        distance_max_ = 0;
        ImageData image_data(imgfrm_queues_len_);
        for (size_t i = 0; i < imgfrm_queues_len_; i++) {
            if (imgfrm_queues_arr_[i].empty()) {
                num_of_empty_++;
                continue;
            }
            if (imgfrm_queues_arr_[i].top().image_time == image_time_target) {
                image_data.put_imgfrm(i, imgfrm_queues_arr_[i].top());
                imgfrm_queues_arr_[i].pop();
            }
            if (imgfrm_queues_arr_[i].empty()) {
                num_of_empty_++;
                continue;
            }
            uint64_t image_time_current = imgfrm_queues_arr_[i].top().image_time;
            if (image_time_current < image_time_min_) {
                image_time_min_ = image_time_current;
            }
            uint64_t distance_current = imgfrm_queues_arr_[i].distance();
            if (distance_current > distance_max_) {
                distance_max_ = distance_current;
            }
        }
        // push image_data into blocking queue
        image_data.event_time = image_time_target;
        image_data.wait_threshold = wait_threshold_;
        if (image_time_target > image_time_last_) {
            image_data.late_arrived = false;
            imgdat_queue_.push(image_data);
            image_time_last_ = image_time_target;
        } else {
            image_data.late_arrived = true;
            imgdat_queue_late_.push(image_data);
        }
        return true;
    } else {
        return false;
    }
}

bool diffraflow::CmbImgCache::take_image(ImageData& image_data) {
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

void diffraflow::CmbImgCache::stop(int wait_time) {
    stopped_ = true;
    lock_guard<mutex> lk(data_mtx_);
    while (do_alignment_(true));
    if (wait_time > 0) {
        unique_lock<mutex> ulk(stop_mtx_);
        stop_cv_.wait_for(ulk, std::chrono::milliseconds(wait_time),
            [this]() {return imgdat_queue_late_.empty() && imgdat_queue_.empty();}
        );
    }
    imgdat_queue_late_.stop();
    imgdat_queue_.stop();
}
