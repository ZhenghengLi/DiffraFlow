#ifndef __IngImgFtrBuffer_H__
#define __IngImgFtrBuffer_H__

#include <cstddef>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <atomic>

#include "ImageDataFeature.hh"
#include "ImageDataField.hh"
#include "ImageFeature.hh"

namespace diffraflow {

    class IngImgFtrBuffer {
    public:
        IngImgFtrBuffer(size_t capacity, bool use_gpu = false);
        ~IngImgFtrBuffer();

        bool mem_ready() const { return mem_ready_; }

        int next();
        void done(int idx);
        void flag(int idx);

        ImageDataField* image_data_host(int idx) { return (ImageDataField*)(buffer_host_ + element_size_ * idx); }
        ImageFeature* image_feature_host(int idx) {
            return (ImageFeature*)(buffer_host_ + element_size_ * idx + feature_offset_);
        }
        ImageDataField* image_data_device(int idx) { return (ImageDataField*)(buffer_device_ + element_size_ * idx); }
        ImageFeature* image_feature_device(int idx) {
            return (ImageFeature*)(buffer_device_ + element_size_ * idx + feature_offset_);
        }

        std::shared_ptr<ImageDataFeature> flag_image();

        void stop();

    private:
        bool use_gpu_;
        bool mem_ready_;

        size_t capacity_;
        size_t element_size_;
        size_t feature_offset_;
        size_t buffer_size_;

        char* buffer_host_;
        char* buffer_device_;

        int head_idx_;
        int tail_idx_;
        int flag_idx_;

        std::mutex range_mtx_;
        std::mutex flag_mtx_;
        std::mutex next_mtx_;
        std::condition_variable next_cv_;

        std::atomic_bool stopped_;
    };
} // namespace diffraflow

#endif
