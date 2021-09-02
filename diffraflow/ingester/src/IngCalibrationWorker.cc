#include "IngCalibrationWorker.hh"
#include "IngImgFtrBuffer.hh"
#include "H5Cpp.h"
#include "Calibration.hh"
#include "cudatools.hh"

log4cxx::LoggerPtr diffraflow::IngCalibrationWorker::logger_ = log4cxx::Logger::getLogger("IngCalibrationWorker");

diffraflow::IngCalibrationWorker::IngCalibrationWorker(
    IngImgFtrBuffer* buffer, IngBufferItemQueue* queue_in, IngBufferItemQueue* queue_out, bool use_gpu, int gpu_index)
    : image_feature_buffer_(buffer), item_queue_in_(queue_in), item_queue_out_(queue_out), use_gpu_(use_gpu),
      gpu_index_(gpu_index) {

    worker_status_ = kNotStart;

    calib_data_host_ = new CalibDataField();
    calib_data_device_ = nullptr;

    // init calibration parameters
    for (size_t m = 0; m < MOD_CNT; m++) {
        for (size_t l = 0; l < GLV_CNT; l++) {
            for (size_t h = 0; h < FRAME_H; h++) {
                for (size_t w = 0; w < FRAME_W; w++) {
                    calib_data_host_->gain[m][l][h][w] = 1.0;
                    calib_data_host_->pedestal[m][l][h][w] = 0.0;
                }
            }
        }
    }

    if (use_gpu_) {
        cudaError_t cuda_err = cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);
        if (cuda_err != cudaSuccess) {
            LOG4CXX_WARN(logger_, "Failed to create cuda stream with error: " << cudaGetErrorString(cuda_err));
        }
    }
}

bool diffraflow::IngCalibrationWorker::read_calib_file(const char* calib_file) {

    H5::Exception::dontPrint();

    try {
        H5::H5File* h5file = new H5::H5File(calib_file, H5F_ACC_RDONLY);

        // gain
        H5::DataSet gain_dset = h5file->openDataSet("gain");
        H5::DataSpace gain_file_space = gain_dset.getSpace();
        hsize_t gain_mem_dim[] = {MOD_CNT, GLV_CNT, FRAME_H, FRAME_W};
        hsize_t gain_offset[] = {0, 0, 0, 0};
        gain_file_space.selectHyperslab(H5S_SELECT_SET, gain_mem_dim, gain_offset);
        H5::DataSpace gain_mem_space(4, gain_mem_dim);
        gain_dset.read(calib_data_host_->gain, H5::PredType::NATIVE_FLOAT, gain_mem_space, gain_file_space);

        // pedestal
        H5::DataSet pedestal_dset = h5file->openDataSet("pedestal");
        H5::DataSpace pedestal_file_space = pedestal_dset.getSpace();
        hsize_t pedestal_mem_dim[] = {MOD_CNT, GLV_CNT, FRAME_H, FRAME_W};
        hsize_t pedestal_offset[] = {0, 0, 0, 0};
        pedestal_file_space.selectHyperslab(H5S_SELECT_SET, pedestal_mem_dim, pedestal_offset);
        H5::DataSpace pedestal_mem_space(4, pedestal_mem_dim);
        pedestal_dset.read(
            calib_data_host_->pedestal, H5::PredType::NATIVE_FLOAT, pedestal_mem_space, pedestal_file_space);

        h5file->close();
        delete h5file;
        h5file = nullptr;

    } catch (H5::Exception& e) {
        LOG4CXX_ERROR(logger_, "found error when reading calibration parameters from HDF5 file " << calib_file << " : "
                                                                                                 << e.getDetailMsg());
        return false;
    } catch (...) {
        LOG4CXX_ERROR(logger_, "found unknown error when read calibration parameters from HDF5 file: " << calib_file);
        return false;
    }

    // change the unit of gain from ADC/keV to keV/ADC.
    for (size_t m = 0; m < MOD_CNT; m++) {
        for (size_t l = 0; l < GLV_CNT; l++) {
            for (size_t h = 0; h < FRAME_H; h++) {
                for (size_t w = 0; w < FRAME_W; w++) {
                    if (calib_data_host_->gain[m][l][h][w] > 0) {
                        calib_data_host_->gain[m][l][h][w] = 1.0 / calib_data_host_->gain[m][l][h][w];
                    } else {
                        calib_data_host_->gain[m][l][h][w] = 1.0;
                        LOG4CXX_ERROR(logger_, "found invalid gain: calib_gain_["
                                                   << m << "][" << l << "][" << h << "][" << w
                                                   << "] = " << calib_data_host_->gain[m][l][h][w]);
                        return false;
                    }
                }
            }
        }
    }

    // allocate memory on GPU and copy calib_data_host_ into it if use GPU
    if (use_gpu_) {
        cudaError_t cuda_err = cudaMalloc(&calib_data_device_, sizeof(CalibDataField));
        if (cuda_err != cudaSuccess) {
            LOG4CXX_ERROR(logger_, "Failed to allocate memory on GPU for calibration parameters with error: "
                                       << cudaGetErrorString(cuda_err));
            return false;
        }
        cuda_err = cudaMemcpyAsync(
            calib_data_device_, calib_data_host_, sizeof(CalibDataField), cudaMemcpyHostToDevice, cuda_stream_);
        if (cuda_err != cudaSuccess) {
            LOG4CXX_ERROR(logger_, "cudaMemcpyAsync failed for copying calibraion parameters into GPU with error: "
                                       << cudaGetErrorString(cuda_err));
            return false;
        }
        cuda_err = cudaStreamSynchronize(cuda_stream_);
        if (cuda_err == cudaSuccess) {
            LOG4CXX_INFO(logger_, "copying calibration parameters into GPU succeeds.");
        } else {
            LOG4CXX_ERROR(
                logger_, "cudaStreamSynchronize failed for copying calibraion parameters into GPU with error: "
                             << cudaGetErrorString(cuda_err));
            return false;
        }
    }

    return true;
}

diffraflow::IngCalibrationWorker::~IngCalibrationWorker() {

    stop();

    delete calib_data_host_;
    calib_data_host_ = nullptr;

    if (use_gpu_) {
        cudaError_t cuda_err = cudaStreamSynchronize(cuda_stream_);
        if (cuda_err != cudaSuccess) {
            LOG4CXX_WARN(logger_, "cudaStreamSynchronize failed with error: " << cudaGetErrorString(cuda_err));
        }
        cuda_err = cudaStreamDestroy(cuda_stream_);
        if (cuda_err != cudaSuccess) {
            LOG4CXX_WARN(logger_, "cudaStreamDestroy failed with error: " << cudaGetErrorString(cuda_err));
        }
        if (calib_data_device_ != nullptr) {
            cuda_err = cudaFree(calib_data_device_);
            if (cuda_err != cudaSuccess) {
                LOG4CXX_WARN(logger_, "cudaFree failed with error: " << cudaGetErrorString(cuda_err));
            }
            calib_data_device_ = nullptr;
        }
    }
}

void diffraflow::IngCalibrationWorker::do_calib_(const shared_ptr<IngBufferItem>& item) {

    char* element_host = image_feature_buffer_->element_host(item->index);
    char* element_device = image_feature_buffer_->element_device(item->index);
    size_t element_size = image_feature_buffer_->element_size();

    ImageDataField* image_data_host = image_feature_buffer_->image_data_host(item->index);
    ImageDataField* image_data_device = image_feature_buffer_->image_data_device(item->index);

    if (use_gpu_) {
        // copy data into GPU
        cudaMemcpyAsync(element_device, element_host, element_size, cudaMemcpyHostToDevice, cuda_stream_);
        // do calib on GPU
        Calibration::do_calib_gpu(cuda_stream_, image_data_device, calib_data_device_);

        // // if feature extraction needs both CPU and GPU: copy calibrated data back to CPU here
        // cudaMemcpyAsync(
        //     image_data_host, image_data_device, sizeof(ImageDataField), cudaMemcpyDeviceToHost, cuda_stream_);

        // wait to finish
        cudaStreamSynchronize(cuda_stream_);
    } else {
        Calibration::do_calib_cpu(image_data_host, calib_data_host_);
    }
}

int diffraflow::IngCalibrationWorker::run_() {

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

        do_calib_(item);

        if (item_queue_out_->push(item)) {
            LOG4CXX_DEBUG(logger_, "pushed the calibrated data into queue.");
        } else {
            break;
        }
    }
    worker_status_ = kStopped;
    cv_status_.notify_all();
    return result;
}

bool diffraflow::IngCalibrationWorker::start() {
    if (!(worker_status_ == kNotStart || worker_status_ == kStopped)) {
        return false;
    }
    worker_status_ = kNotStart;
    worker_ = async(std::launch::async, &IngCalibrationWorker::run_, this);
    unique_lock<mutex> ulk(mtx_status_);
    cv_status_.wait(ulk, [this]() { return worker_status_ != kNotStart; });
    if (worker_status_ == kRunning) {
        return true;
    } else {
        return false;
    }
}

void diffraflow::IngCalibrationWorker::wait() {
    if (worker_.valid()) {
        worker_.wait();
    }
}

int diffraflow::IngCalibrationWorker::stop() {
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
