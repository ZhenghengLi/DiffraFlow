#include "IngImgFtrBuffer.hh"
#include "ImageDataField.hh"
#include "ImageFeature.hh"
#include "ImageDataFeature.hh"

#include <cuda_runtime.h>

using std::lock_guard;
using std::unique_lock;
using std::mutex;
using std::shared_ptr;
using std::make_shared;

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
        buffer_host_ = (char*)malloc(buffer_size_);
    }
    head_idx_ = -1;
    tail_idx_ = -1;
    flag_idx_ = -1;
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

int diffraflow::IngImgFtrBuffer::next() {
    unique_lock<mutex> ulk(range_mtx_);
    lock_guard<mutex> lg(flag_mtx_);
    int next_head = head_idx_ + 1;
    if (next_head == capacity_) next_head = 0;
    next_cv_.wait(ulk, [&] { return next_head != tail_idx_; });
    head_idx_ = next_head;
    if (head_idx_ == flag_idx_) {
        flag_idx_ = -1;
    }
    return head_idx_;
}

void diffraflow::IngImgFtrBuffer::done(int idx) {
    lock_guard<mutex> lg(range_mtx_);
    if (idx >= 0 && idx < capacity_) {
        tail_idx_ = idx;
        next_cv_.notify_all();
    }
}

void diffraflow::IngImgFtrBuffer::flag(int idx) {
    lock_guard<mutex> lg(flag_mtx_);
    if (idx >= 0 && idx < capacity_) {
        flag_idx_ = idx;
    }
}

shared_ptr<diffraflow::ImageDataFeature> diffraflow::IngImgFtrBuffer::flag_image() {
    lock_guard<mutex> lg(flag_mtx_);
    if (flag_idx_ >= 0) {
        return make_shared<ImageDataFeature>(image_data_host(flag_idx_), image_feature_host(flag_idx_));
    } else {
        return nullptr;
    }
}