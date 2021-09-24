#include "IngFeatureExtracter.hh"
#include "IngImgFtrBuffer.hh"
#include "IngConfig.hh"
#include "cudatools.hh"

log4cxx::LoggerPtr diffraflow::IngFeatureExtracter::logger_ = log4cxx::Logger::getLogger("IngFeatureExtracter");

diffraflow::IngFeatureExtracter::IngFeatureExtracter(IngImgFtrBuffer* buffer, IngBufferItemQueue* queue_in,
    IngBufferItemQueue* queue_out, IngConfig* conf_obj, bool use_gpu, int gpu_index)
    : image_feature_buffer_(buffer), item_queue_in_(queue_in), item_queue_out_(queue_out), config_obj_(conf_obj),
      use_gpu_(use_gpu), gpu_index_(gpu_index) {
    worker_status_ = kNotStart;

    mean_rms_sum_device_ = nullptr;
    mean_rms_count_device_ = nullptr;

    if (use_gpu_) {
        // create CUDA streams
        cudaError_t cuda_err = cudaStreamCreateWithFlags(&cuda_stream_peak_msse_, cudaStreamNonBlocking);
        if (cuda_err != cudaSuccess) {
            LOG4CXX_ERROR(logger_, "Failed to create cuda stream with error: " << cudaGetErrorString(cuda_err));
        }
        cuda_err = cudaStreamCreateWithFlags(&cuda_stream_mean_rms_, cudaStreamNonBlocking);
        if (cuda_err != cudaSuccess) {
            LOG4CXX_ERROR(logger_, "Failed to create cuda stream with error: " << cudaGetErrorString(cuda_err));
        }
        // allocate memory on GPU
        cuda_err = cudaMalloc(&mean_rms_sum_device_, sizeof(double));
        if (cuda_err != cudaSuccess) {
            LOG4CXX_ERROR(logger_, "cudaMalloc failed with error: " << cudaGetErrorString(cuda_err));
        }
        cuda_err = cudaMalloc(&mean_rms_count_device_, sizeof(int));
        if (cuda_err != cudaSuccess) {
            LOG4CXX_ERROR(logger_, "cudaMalloc failed with error: " << cudaGetErrorString(cuda_err));
        }
    }
}

diffraflow::IngFeatureExtracter::~IngFeatureExtracter() {

    stop();

    if (use_gpu_) {
        // sync
        cudaError_t cuda_err = cudaStreamSynchronize(cuda_stream_peak_msse_);
        if (cuda_err != cudaSuccess) {
            LOG4CXX_ERROR(logger_, "cudaStreamSynchronize failed with error: " << cudaGetErrorString(cuda_err));
        }
        cuda_err = cudaStreamSynchronize(cuda_stream_mean_rms_);
        if (cuda_err != cudaSuccess) {
            LOG4CXX_ERROR(logger_, "cudaStreamSynchronize failed with error: " << cudaGetErrorString(cuda_err));
        }
        // destroy
        cuda_err = cudaStreamDestroy(cuda_stream_peak_msse_);
        if (cuda_err != cudaSuccess) {
            LOG4CXX_ERROR(logger_, "cudaStreamDestroy failed with error: " << cudaGetErrorString(cuda_err));
        }
        cuda_err = cudaStreamDestroy(cuda_stream_mean_rms_);
        if (cuda_err != cudaSuccess) {
            LOG4CXX_ERROR(logger_, "cudaStreamDestroy failed with error: " << cudaGetErrorString(cuda_err));
        }
        // free memory
        cuda_err = cudaFree(mean_rms_sum_device_);
        if (cuda_err != cudaSuccess) {
            LOG4CXX_ERROR(logger_, "cudaFree failed with error: " << cudaGetErrorString(cuda_err));
        }
        cuda_err = cudaFree(mean_rms_count_device_);
        if (cuda_err != cudaSuccess) {
            LOG4CXX_ERROR(logger_, "cudaFree failed with error: " << cudaGetErrorString(cuda_err));
        }
    }
}

void diffraflow::IngFeatureExtracter::extract_feature_(const shared_ptr<IngBufferItem>& item) {
    lock_guard<mutex> common_variables_lg(common_variables_mtx_);
    // some example code
    image_feature_buffer_->image_feature_host(item->index)->global_mean = 100;
    image_feature_buffer_->image_feature_host(item->index)->global_rms = 20;
    image_feature_buffer_->image_feature_host(item->index)->peak_pixels = 10;
}

int diffraflow::IngFeatureExtracter::run_() {

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
