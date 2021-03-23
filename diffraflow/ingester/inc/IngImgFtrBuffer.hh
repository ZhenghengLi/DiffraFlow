#ifndef __IngImgFtrBuffer_H__
#define __IngImgFtrBuffer_H__

#include <cstddef>

namespace diffraflow {

    class IngImgFtrBuffer {
    public:
        IngImgFtrBuffer(size_t capacity, bool use_gpu = false);
        ~IngImgFtrBuffer();

        bool mem_ready() const { return mem_ready_; }

    private:
        bool use_gpu_;
        bool mem_ready_;

        size_t capacity_;
        size_t element_size_;
        size_t feature_offset_;
        size_t buffer_size_;

        void* buffer_host_;
        void* buffer_device_;

        int head_;
        int tail_;
        int flag_;
    };
} // namespace diffraflow

#endif