#ifndef __ByteBuffer_H__
#define __ByteBuffer_H__

#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace diffraflow {
    class ByteBuffer {
    public:
        ByteBuffer() : data_(nullptr), size_(0) {}

        explicit ByteBuffer(size_t len) : size_(len) {
            if (size_ > 0) {
                data_ = (char*)malloc(size_);
            } else {
                data_ = nullptr;
            }
        }

        ByteBuffer(const ByteBuffer& buffer) : data_(nullptr) { copy_(buffer); }
        ~ByteBuffer() { free(data_); }

        ByteBuffer& operator=(const ByteBuffer& buffer) {
            copy_(buffer);
            return *this;
        }

        char* data() { return data_; }
        size_t size() { return size_; }
        void resize(size_t len) {
            size_ = len;
            if (size_ > 0) {
                data_ = (char*)realloc(data_, size_);
            } else {
                free(data_);
                data_ = nullptr;
            }
        }

    private:
        void copy_(const ByteBuffer& buffer) {
            free(data_);
            size_ = buffer.size_;
            if (size_ > 0) {
                data_ = (char*)malloc(size_);
                memcpy(data_, buffer.data_, buffer.size_);
            } else {
                data_ = nullptr;
            }
        }

    private:
        char* data_;
        size_t size_;
    };
} // namespace diffraflow

#endif