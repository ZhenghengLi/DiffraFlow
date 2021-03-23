#include "IngImgFtrBuffer.hh"
#include "ImageDataField.hh"
#include "ImageFeature.hh"

#include <cuda_runtime.h>

diffraflow::IngImgFtrBuffer::IngImgFtrBuffer(size_t capacity, bool use_gpu) : capacity_(capacity), use_gpu_(use_gpu) {
    mem_ready_ = true;
    element_size_ = sizeof(ImageDataField) + sizeof(ImageFeature);
    feature_offset_ = sizeof(ImageDataField);
    buffer_size_ = element_size_ * capacity_;
    buffer_host_ = nullptr;
    buffer_device_ = nullptr;
    if (use_gpu_) {
        if (cudaSuccess != cudaMallocHost(&buffer_host_, buffer_size_)) mem_ready_ = false;
        if (cudaSuccess != cudaMalloc(&buffer_device_, buffer_size_)) mem_ready_ = false;
    } else {
        buffer_host_ = malloc(buffer_size_);
    }
    head_ = -1;
    tail_ = -1;
    flag_ = -1;
}

diffraflow::IngImgFtrBuffer::~IngImgFtrBuffer() {
    if (use_gpu_) {
        cudaFreeHost(buffer_host_);
        buffer_host_ = nullptr;
        cudaFree(buffer_device_);
        buffer_device_ = nullptr;
    } else {
        free(buffer_host_);
        buffer_host_ = nullptr;
    }
}
