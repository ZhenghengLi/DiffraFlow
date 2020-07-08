#include "CmbImgCache.hh"
#include "ImageFramePtr.hh"
#include "ImageData.hh"

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <limits>
#include <chrono>

using std::numeric_limits;
using std::lock_guard;
using std::unique_lock;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::system_clock;
using std::milli;

log4cxx::LoggerPtr diffraflow::CmbImgCache::logger_ = log4cxx::Logger::getLogger("CmbImgCache");

diffraflow::CmbImgCache::CmbImgCache(size_t num_of_dets, size_t img_q_ms, int max_lt) {
    imgfrm_queues_len_ = num_of_dets;
    imgfrm_queues_arr_ = new TimeOrderedQueue<ImageFramePtr, int64_t>[imgfrm_queues_len_];
    imgdat_queue_.set_maxsize(img_q_ms);
    // wait forever
    wait_threshold_ = numeric_limits<int64_t>::max();
    image_time_min_ = numeric_limits<uint64_t>::max();
    image_time_last_ = 0;
    num_of_empty_ = imgfrm_queues_len_;
    distance_max_ = 0;
    stopped_ = false;

    max_linger_time_ = max_lt;
    clear_flag_ = true;
    latest_push_time_ = 0;

    clear_worker_ = new thread([this]() {
        while (!stopped_) {
            unique_lock<mutex> ulk(data_mtx_);
            if (clear_flag_) {
                clear_cv_.wait(ulk, [&]() {return stopped_ || !clear_flag_;});
                if (stopped_) break;
            }
            duration<double, milli> current_time = system_clock::now().time_since_epoch();
            double linger_time = current_time.count() - latest_push_time_;
            if (linger_time >= max_linger_time_) {
                clear_cache_();
                clear_flag_ = true;
            } else {
                int wait_time = max_linger_time_ - linger_time;
                clear_cv_.wait_for(ulk, milliseconds(wait_time), [&]() {return stopped_.load();});
                if (stopped_) break;
            }
        }
    });

    alignment_metrics.total_pushed_frames = 0;
    alignment_metrics.total_aligned_images = 0;
    alignment_metrics.total_late_arrived = 0;
    alignment_metrics.total_partial_images = 0;
}

diffraflow::CmbImgCache::~CmbImgCache() {
    delete[] imgfrm_queues_arr_;
    stop(0);
    clear_worker_->join();
    delete clear_worker_;
}

bool diffraflow::CmbImgCache::push_frame(const ImageFramePtr& image_frame) {
    if (stopped_) return false;

    if (image_frame->detector_id < 0 || image_frame->detector_id >= (int)imgfrm_queues_len_) {
        LOG4CXX_WARN(logger_, "Detector ID is out of range: " << image_frame->detector_id);
        return false;
    }

    lock_guard<mutex> lk(data_mtx_);

    if (image_frame->image_time < image_time_min_) {
        image_time_min_ = image_frame->image_time;
    }
    if (imgfrm_queues_arr_[image_frame->detector_id].empty()) {
        num_of_empty_--;
    }
    imgfrm_queues_arr_[image_frame->detector_id].push(image_frame);
    int64_t distance_current = imgfrm_queues_arr_[image_frame->detector_id].distance();
    if (distance_current > distance_max_) {
        distance_max_ = distance_current;
    }

    alignment_metrics.total_pushed_frames++;

    while (true) {
        shared_ptr<ImageData> image_data = make_shared<ImageData>(imgfrm_queues_len_);
        if (!do_alignment_(image_data, false)) {
            break;
        }
        if (image_data->event_time < image_time_last_) {
            image_data->late_arrived = true;
            alignment_metrics.total_late_arrived++;
        } else {
            image_data->late_arrived = false;
            image_time_last_ = image_data->event_time;
        }
        alignment_metrics.total_aligned_images++;
        for (const bool& item : image_data->alignment_vec) {
            if (!item) {
                alignment_metrics.total_partial_images++;
                break;
            }
        }
        // for debug
        image_data->print();
        LOG4CXX_DEBUG(logger_, "before push into imgdat_queue_.");
        if (imgdat_queue_.push(image_data)) {
            LOG4CXX_DEBUG(logger_, "pushed one image into imgdat_queue_.");
        } else {
            LOG4CXX_INFO(logger_, "failed to push image data, as imgdat_queue_ is stopped.");
            return false;
        }
    }

    duration<double, milli> current_time = system_clock::now().time_since_epoch();
    latest_push_time_ = current_time.count();
    if (clear_flag_) {
        clear_flag_ = false;
        clear_cv_.notify_all();
    }

    return true;
}

bool diffraflow::CmbImgCache::do_alignment_(shared_ptr<ImageData> image_data, bool force_flag) {
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
            if (imgfrm_queues_arr_[i].top()->image_time == image_time_target) {
                image_data->put_imgfrm(i, *imgfrm_queues_arr_[i].top());
                imgfrm_queues_arr_[i].pop();
            }
            if (imgfrm_queues_arr_[i].empty()) {
                num_of_empty_++;
                continue;
            }
            uint64_t image_time_current = imgfrm_queues_arr_[i].top()->image_time;
            if (image_time_current < image_time_min_) {
                image_time_min_ = image_time_current;
            }
            int64_t distance_current = imgfrm_queues_arr_[i].distance();
            if (distance_current > distance_max_) {
                distance_max_ = distance_current;
            }
        }
        image_data->event_time = image_time_target;
        image_data->wait_threshold = wait_threshold_;
        image_data->set_defined();
        return true;
    } else {
        return false;
    }
}

bool diffraflow::CmbImgCache::take_image(shared_ptr<ImageData>& image_data) {
    bool result = imgdat_queue_.take(image_data);
    if (stopped_ && imgdat_queue_.empty()) {
        stop_cv_.notify_all();
    }
    return result;
}

void diffraflow::CmbImgCache::clear_cache_() {
    while (true) {
        shared_ptr<ImageData> image_data = make_shared<ImageData>(imgfrm_queues_len_);
        if (!do_alignment_(image_data, true)) {
            break;
        }
        if (image_data->event_time < image_time_last_) {
            image_data->late_arrived = true;
        } else {
            image_data->late_arrived = false;
            image_time_last_ = image_data->event_time;
        }
        // for debug
        image_data->print();
        LOG4CXX_DEBUG(logger_, "before offer into imgdat_queue_.");
        if (imgdat_queue_.offer(image_data)) {
            LOG4CXX_DEBUG(logger_, "offerred one image into imgdat_queue_.");
        } else {
            LOG4CXX_INFO(logger_, "failed to offer image data, as imgdat_queue_ is full.");
            break;
        }
    }
}

void diffraflow::CmbImgCache::stop(int wait_time) {
    stopped_ = true;

    imgdat_queue_.stop();

    lock_guard<mutex> lk(data_mtx_);

    // clear all data in imgfrm_queues_arr_
    clear_cache_();

    // wait ingester to consume image data in queue for wait_time
    if (wait_time > 0) {
        unique_lock<mutex> ulk(stop_mtx_);
        stop_cv_.wait_for(ulk, std::chrono::milliseconds(wait_time), [this]() { return imgdat_queue_.empty(); });
    }

    clear_cv_.notify_all();
}

json::value diffraflow::CmbImgCache::collect_metrics() {

    json::value alignment_metrics_json;
    alignment_metrics_json["total_pushed_frames"] = json::value::number(alignment_metrics.total_pushed_frames.load());
    alignment_metrics_json["total_aligned_images"] = json::value::number(alignment_metrics.total_aligned_images.load());
    alignment_metrics_json["total_late_arrived"] = json::value::number(alignment_metrics.total_late_arrived.load());
    alignment_metrics_json["total_partial_images"] = json::value::number(alignment_metrics.total_partial_images.load());

    json::value queue_metrics_json;
    {
        lock_guard<mutex> lg(data_mtx_);
        queue_metrics_json["image_data_queue_size"] = json::value::number((uint32_t)imgdat_queue_.size());
        json::value image_frame_queue_sizes;
        for (size_t i = 0; i < imgfrm_queues_len_; i++) {
            image_frame_queue_sizes[i] = json::value::number((uint32_t)imgfrm_queues_arr_[i].size());
        }
        queue_metrics_json["image_frame_queue_sizes"] = image_frame_queue_sizes;
    }

    json::value root_json;
    root_json["alignment_stats"] = alignment_metrics_json;
    root_json["queue_stats"] = queue_metrics_json;

    return root_json;
}
