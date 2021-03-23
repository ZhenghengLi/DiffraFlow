#include "ImageWithFeature.hh"

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <cuda_runtime.h>

log4cxx::LoggerPtr diffraflow::ImageWithFeature::logger_ = log4cxx::Logger::getLogger("ImageWithFeature");

diffraflow::ImageWithFeature::ImageWithFeature(bool use_gpu) : use_gpu_(use_gpu) {
    ref_cnt_ptr_ = new atomic_int(1);
    mem_ready_ = true;

    image_data_host_ptr_ = nullptr;
    image_feature_host_ptr_ = nullptr;
    image_data_device_ptr_ = nullptr;
    image_feature_device_ptr_ = nullptr;

    if (use_gpu_) {
        // host
        if (cudaSuccess != cudaMallocHost(&image_data_host_ptr_, sizeof(ImageDataField))) mem_ready_ = false;
        if (cudaSuccess != cudaMallocHost(&image_feature_host_ptr_, sizeof(ImageFeature))) mem_ready_ = false;
        // device
        if (cudaSuccess != cudaMalloc(&image_data_device_ptr_, sizeof(ImageDataField))) mem_ready_ = false;
        if (cudaSuccess != cudaMalloc(&image_feature_device_ptr_, sizeof(ImageFeature))) mem_ready_ = false;
    } else {
        image_data_host_ptr_ = new ImageDataField();
        image_feature_host_ptr_ = new ImageFeature();
    }
}

diffraflow::ImageWithFeature::ImageWithFeature(const ImageWithFeature& obj) { copyObj_(obj); }

diffraflow::ImageWithFeature& diffraflow::ImageWithFeature::operator=(const ImageWithFeature& right) {
    copyObj_(right);
    return *this;
}

void diffraflow::ImageWithFeature::copyObj_(const ImageWithFeature& obj) {
    if (this == &obj) return;

    *this = obj;
    (*this->ref_cnt_ptr_)++;
}

diffraflow::ImageWithFeature::~ImageWithFeature() {
    (*this->ref_cnt_ptr_)--;

    if (*this->ref_cnt_ptr_ < 1) {

        // delete the reference counter
        delete ref_cnt_ptr_;
        ref_cnt_ptr_ = nullptr;

        // delete the data on both host and device
        if (use_gpu_) {
            // host
            if (image_data_host_ptr_) {
                cudaFreeHost(image_data_host_ptr_);
                image_data_host_ptr_ = nullptr;
            }
            if (image_feature_host_ptr_) {
                cudaFreeHost(image_feature_host_ptr_);
                image_feature_host_ptr_ = nullptr;
            }
            // device
            if (image_data_device_ptr_) {
                cudaFree(image_data_device_ptr_);
                image_data_device_ptr_ = nullptr;
            }
            if (image_feature_device_ptr_) {
                cudaFree(image_feature_device_ptr_);
                image_feature_device_ptr_ = nullptr;
            }
        } else {
            delete image_data_host_ptr_;
            image_data_host_ptr_ = nullptr;
            delete image_feature_host_ptr_;
            image_feature_host_ptr_ = nullptr;
        }
    }
}
