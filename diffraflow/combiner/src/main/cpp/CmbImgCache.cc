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
    delete [] imgfrm_queues_arr_;
    stop(0);
}

bool diffraflow::CmbImgCache::push_frame(const ImageFrame& image_frame) {
    if (stopped_) return false;

    if (image_frame.detector_id < 0 || image_frame.detector_id >= (int) imgfrm_queues_len_) {
        LOG4CXX_WARN(logger_, "Detector ID is out of range: " << image_frame.detector_id);
        return false;
    }

    lock_guard<mutex> lk(data_mtx_);

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

    while (true) {
        ImageData image_data(imgfrm_queues_len_);
        if (!do_alignment_(image_data, false)) {
            break;
        }
        if (image_data.event_time < image_time_last_) {
            image_data.late_arrived = true;
        } else {
            image_data.late_arrived = false;
            image_time_last_ = image_data.event_time;
        }
        // for debug
        image_data.print();
        LOG4CXX_DEBUG(logger_, "before push into imgdat_queue_.");
        if (imgdat_queue_.push(image_data)) {
            LOG4CXX_DEBUG(logger_, "pushed one image into imgdat_queue_.");
        } else {
            LOG4CXX_INFO(logger_, "failed to push image data, as imgdat_queue_ is stopped.");
            return false;
        }
    }

    return true;
}

bool diffraflow::CmbImgCache::do_alignment_(ImageData& image_data, bool force_flag) {
    if (num_of_empty_ == imgfrm_queues_len_) {
        return false;
    }
    if (num_of_empty_ <= 0 || distance_max_ > wait_threshold_ || force_flag) {
        uint64_t image_time_target = image_time_min_;
        image_time_min_ = numeric_limits<uint64_t>::max();
        num_of_empty_ = 0;
        distance_max_ = 0;
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
        image_data.event_time = image_time_target;
        image_data.wait_threshold = wait_threshold_;
        return true;
    } else {
        return false;
    }
}

bool diffraflow::CmbImgCache::take_image(ImageData& image_data) {
    bool result = imgdat_queue_.take(image_data);
    if (stopped_ && imgdat_queue_.empty()) {
        stop_cv_.notify_all();
    }
    return result;
}

void diffraflow::CmbImgCache::stop(int wait_time) {
    stopped_ = true;

    imgdat_queue_.stop();

    lock_guard<mutex> lk(data_mtx_);

    // clear all data in imgfrm_queues_arr_
    while (true) {
        ImageData image_data(imgfrm_queues_len_);
        if (!do_alignment_(image_data, true)) {
            break;
        }
        if (image_data.event_time < image_time_last_) {
            image_data.late_arrived = true;
        } else {
            image_data.late_arrived = false;
            image_time_last_ = image_data.event_time;
        }
        // for debug
        image_data.print();
        LOG4CXX_DEBUG(logger_, "before offer into imgdat_queue_.");
        if (imgdat_queue_.offer(image_data)) {
            LOG4CXX_DEBUG(logger_, "offerred one image into imgdat_queue_.");
        } else {
            LOG4CXX_INFO(logger_, "failed to offer image data, as imgdat_queue_ is full.");
            break;
        }
    }

    // wait ingester to consume image data in queue for wait_time
    if (wait_time > 0) {
        unique_lock<mutex> ulk(stop_mtx_);
        stop_cv_.wait_for(ulk, std::chrono::milliseconds(wait_time),
            [this]() {return imgdat_queue_.empty();}
        );
    }

}
