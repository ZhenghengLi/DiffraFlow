#ifndef __ImageWithFeature_H__
#define __ImageWithFeature_H__

#include "ImageDataType.hh"
#include "ImageFeature.hh"

#include <vector>
#include <memory>
#include <log4cxx/logger.h>

using std::vector;
using std::shared_ptr;

namespace diffraflow {
    class ImageWithFeature {
        // copies of the same instance of this class share the same internal data.
        // the internal data will be automatically deleted when all copies of the instance are destroyied.

    public:
        ImageWithFeature(bool use_gpu = false);
        ImageWithFeature(const ImageWithFeature& obj);
        ~ImageWithFeature();

        ImageWithFeature& operator=(const ImageWithFeature& right);

    public:
        // raw image data
        shared_ptr<vector<char>> image_data_raw;

        // internal pointer getters
        //// host
        ImageDataField* image_data_host() const { return image_data_host_ptr_; };
        ImageFeature* image_feature_host() const { return image_feature_host_ptr_; };
        //// device
        ImageDataField* image_data_device() const { return image_data_device_ptr_; };
        ImageFeature* image_feature_device() const { return image_feature_device_ptr_; };

        bool mem_ready() { return mem_ready_; }

    private:
        void copyObj_(const ImageWithFeature& obj);

    private:
        bool use_gpu_;
        int* ref_cnt_ptr_;
        bool mem_ready_;

        // self-managed pointers
        //// host
        ImageDataField* image_data_host_ptr_;
        ImageFeature* image_feature_host_ptr_;
        //// device
        ImageDataField* image_data_device_ptr_;
        ImageFeature* image_feature_device_ptr_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif