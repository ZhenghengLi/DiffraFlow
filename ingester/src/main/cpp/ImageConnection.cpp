#include "ImageConnection.hpp"

shine::ImageConnection::ImageConnection() {
    buffer_size_ = 1024 * 1024;
    buffer_ = new char[buffer_size_];

}

shine::ImageConnection::~ImageConnection() {
    delete [] buffer_;
    buffer_ = nullptr;
}
