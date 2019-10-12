#include "ImageConnection.hpp"
#include "ImageCache.hpp"

#include <unistd.h>

using std::cout;
using std::cerr;
using std::endl;

shine::ImageConnection::ImageConnection(int sock_fd, ImageCache* img_cache_) {
    buffer_size_ = 1024 * 1024;
    buffer_ = new char[buffer_size_];
    client_sock_fd_ = sock_fd;
    image_cache_ = img_cache_;
}

shine::ImageConnection::~ImageConnection() {
    delete [] buffer_;
    buffer_ = nullptr;
    done_flag_ = false;
}

void shine::ImageConnection::run() {
    if (start_connection()) {
        while (done_flag_) transfering();
    } else {
        close(client_sock_fd_);
        done_flag_ = true;
        return;
    }
}

bool shine::ImageConnection::done() {
    return done_flag_;
}

bool shine::ImageConnection::start_connection() {

    return true;
}

void shine::ImageConnection::transfering() {

    // push data into image_cache_

}