#include "IngImageFilter.hh"
#include "IngImgFtrBuffer.hh"
#include "IngConfig.hh"
#include "ImageFeature.hh"
#include "cudatools.hh"

log4cxx::LoggerPtr diffraflow::IngImageFilter::logger_ = log4cxx::Logger::getLogger("IngImageFilter");

diffraflow::IngImageFilter::IngImageFilter(IngImgFtrBuffer* buffer, IngBufferItemQueue* queue_in,
    IngBufferItemQueue* queue_out, IngConfig* conf_obj, bool use_gpu, int gpu_index)
    : image_feature_buffer_(buffer), item_queue_in_(queue_in), item_queue_out_(queue_out), config_obj_(conf_obj),
      use_gpu_(use_gpu), gpu_index_(gpu_index) {

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

bool diffraflow::IngImageFilter::check_for_save_(const ImageFeature* image_feature) {
    return image_feature->global_mean >= config_obj_->get_dy_saving_global_mean_thr() &&
           image_feature->global_rms >= config_obj_->get_dy_saving_global_rms_thr() &&
           image_feature->peak_pixels >= config_obj_->get_dy_saving_peak_pixels_thr();
}

bool diffraflow::IngImageFilter::check_for_monitor_(const ImageFeature* image_feature) {
    return image_feature->global_mean >= config_obj_->get_dy_monitor_global_mean_thr() &&
           image_feature->global_rms >= config_obj_->get_dy_monitor_global_rms_thr() &&
           image_feature->peak_pixels >= config_obj_->get_dy_monitor_peak_pixels_thr();
}

void diffraflow::IngImageFilter::do_filter(shared_ptr<IngBufferItem>& item) {

    ImageFeature* image_feature_host = image_feature_buffer_->image_feature_host(item->index);
    ImageFeature* image_feature_device = image_feature_buffer_->image_feature_device(item->index);
    ImageDataField* image_data_host = image_feature_buffer_->image_data_host(item->index);
    ImageDataField* image_data_device = image_feature_buffer_->image_data_device(item->index);

    if (use_gpu_) {
        // copy image feature from GPU to CPU
        cudaMemcpyAsync(
            image_feature_host, image_feature_device, sizeof(ImageFeature), cudaMemcpyDeviceToHost, cuda_stream_);
        // wait to finish
        cudaStreamSynchronize(cuda_stream_);

        // // if feature extraction needs both CPU and GPU:
        // cudaMemcpyAsync(
        //     image_feature_host_gpu, image_feature_device, sizeof(ImageFeature), cudaMemcpyDeviceToHost,
        //     cuda_stream_);
        // cudaStreamSynchronize(cuda_stream_);
        // merge_feature_(image_feature_host, image_feature_host_gpu);
    }

    bool monitor = check_for_monitor_(image_feature_host);
    bool save = check_for_save_(image_feature_host);

    // if calibrated data has not been copied back to CPU: copy it here
    if (use_gpu_ && (monitor || save)) {
        // copy image data from GPU to CPU
        cudaMemcpyAsync(
            image_data_host, image_data_device, sizeof(ImageDataField), cudaMemcpyDeviceToHost, cuda_stream_);
        // wait to finish
        cudaStreamSynchronize(cuda_stream_);
    }

    if (monitor) {
        filter_metrics.total_images_for_monitor++;
        image_feature_buffer_->flag(item->index);
    }

    if (save) {
        filter_metrics.total_images_for_save++;
        item->save = true;
    } else {
        item->save = false;
    }
}

int diffraflow::IngImageFilter::run_() {

    if (use_gpu_) {
        cudaError_t cuda_err = cudaSetDevice(gpu_index_);
        if (cuda_err == cudaSuccess) {
            LOG4CXX_INFO(logger_, "Successfully selected " << cudatools::get_device_string(gpu_index_));
        } else {
            LOG4CXX_ERROR(logger_, "Failed to select GPU of device index " << gpu_index_);
            worker_status_ = kStopped;
            cv_status_.notify_all();
            return -1;
        }
    }

    int result = 0;
    worker_status_ = kRunning;
    cv_status_.notify_all();
    shared_ptr<IngBufferItem> item;
    while (worker_status_ != kStopped && item_queue_in_->take(item)) {

        do_filter(item);

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