#include "IngImageFilter.hh"
#include "IngImgFtrBuffer.hh"
#include "IngConfig.hh"
#include "ImageFeature.hh"

log4cxx::LoggerPtr diffraflow::IngImageFilter::logger_ = log4cxx::Logger::getLogger("IngImageFilter");

diffraflow::IngImageFilter::IngImageFilter(IngImgFtrBuffer* buffer, IngBufferItemQueue* queue_in,
    IngBufferItemQueue* queue_out, IngConfig* conf_obj, bool use_gpu)
    : image_feature_buffer_(buffer), item_queue_in_(queue_in), item_queue_out_(queue_out), config_obj_(conf_obj),
      use_gpu_(use_gpu) {

    worker_status_ = kNotStart;

    filter_metrics.total_images_for_monitor = 0;
    filter_metrics.total_images_for_save = 0;
    filter_metrics.total_images_for_save_fail = 0;
    filter_metrics.total_processed_images = 0;

    if (use_gpu_) {
        cudaError_t cuda_err = cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);
        if (cuda_err != cudaSuccess) {
            LOG4CXX_WARN(logger_, "Failed to create cuda stream with error: " << cudaGetErrorString(cuda_err));
        }
    }
}

diffraflow::IngImageFilter::~IngImageFilter() {
    if (use_gpu_) {
        cudaError_t cuda_err = cudaStreamSynchronize(cuda_stream_);
        if (cuda_err != cudaSuccess) {
            LOG4CXX_WARN(logger_, "cudaStreamSynchronize failed with error: " << cudaGetErrorString(cuda_err));
        }
        cuda_err = cudaStreamDestroy(cuda_stream_);
        if (cuda_err != cudaSuccess) {
            LOG4CXX_WARN(logger_, "cudaStreamDestroy failed with error: " << cudaGetErrorString(cuda_err));
        }
    }
}

bool diffraflow::IngImageFilter::check_for_save_(const ImageFeature& image_feature) { return true; }

bool diffraflow::IngImageFilter::check_for_monitor_(const ImageFeature& image_feature) { return true; }

int diffraflow::IngImageFilter::run_() {
    int result = 0;
    worker_status_ = kRunning;
    cv_status_.notify_all();
    shared_ptr<IngBufferItem> item;
    while (worker_status_ != kStopped && item_queue_in_->take(item)) {

        // copy image feature from gpu to cpu if gpu is enabled

        if (check_for_monitor_(*image_feature_buffer_->image_feature_host(item->index))) {
            filter_metrics.total_images_for_monitor++;
            image_feature_buffer_->flag(item->index);
        }

        if (check_for_save_(*image_feature_buffer_->image_feature_host(item->index))) {
            filter_metrics.total_images_for_save++;
            item->save = true;
        } else {
            item->save = false;
        }

        if (item_queue_out_->offer(item)) {
            LOG4CXX_DEBUG(logger_, "successfully pushed one good image into queue for saving.");
        } else {
            LOG4CXX_DEBUG(logger_, "failed to push one good image into queue for saving.");
            filter_metrics.total_images_for_save_fail++;
        }

        filter_metrics.total_processed_images++;
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
    worker_ = async(std::launch::async, &IngImageFilter::run_, this);
    unique_lock<mutex> ulk(mtx_status_);
    cv_status_.wait(ulk, [this]() { return worker_status_ != kNotStart; });
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

json::value diffraflow::IngImageFilter::collect_metrics() {
    json::value filter_metrics_json;
    filter_metrics_json["total_processed_images"] = json::value::number(filter_metrics.total_processed_images.load());
    filter_metrics_json["total_images_for_save"] = json::value::number(filter_metrics.total_images_for_save.load());
    filter_metrics_json["total_images_for_save_fail"] =
        json::value::number(filter_metrics.total_images_for_save_fail.load());
    filter_metrics_json["total_images_for_monitor"] =
        json::value::number(filter_metrics.total_images_for_monitor.load());
    return filter_metrics_json;
}