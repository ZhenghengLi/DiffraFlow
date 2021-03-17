#include "ImageWithFeature.hh"

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

log4cxx::LoggerPtr diffraflow::ImageWithFeature::logger_ = log4cxx::Logger::getLogger("ImageWithFeature");

diffraflow::ImageWithFeature::ImageWithFeature(bool use_gpu) : use_gpu_(use_gpu) {
    ref_cnt_ptr_ = new int(1);
    mem_ready_ = true;

    if (use_gpu_) {
        // cudaHostAlloc
        // cudaMalloc
        // if malloc failed: mem_ready_ = false

        // host
        image_data_host_ptr_ = nullptr;
        image_feature_host_ptr_ = nullptr;
        // device
        image_data_device_ptr_ = nullptr;
        image_feature_device_ptr_ = nullptr;
    } else {
        // host
        image_data_host_ptr_ = new ImageDataField();
        image_feature_host_ptr_ = new ImageFeature();
        // device
        image_data_device_ptr_ = nullptr;
        image_feature_device_ptr_ = nullptr;
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
    *this->ref_cnt_ptr_ += 1;
}

diffraflow::ImageWithFeature::~ImageWithFeature() {
    *this->ref_cnt_ptr_ -= 1;

    if (*this->ref_cnt_ptr_ < 1) {

        // delete the reference counter
        delete ref_cnt_ptr_;
        ref_cnt_ptr_ = nullptr;

        // delete the data on both host and device
        if (use_gpu_) {
            // cudaFreeHost
            // cudaFree
        } else {
            delete image_data_host_ptr_;
            image_data_host_ptr_ = nullptr;
            delete image_feature_host_ptr_;
            image_feature_host_ptr_ = nullptr;
        }
    }
}
