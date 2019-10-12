#include "ImageConnection.hpp"
#include "ImageCacheServer.hpp"

shine::ImageConnection::ImageConnection(int sock_fd, ImageCacheServer* server) {
    buffer_size_ = 1024 * 1024;
    buffer_ = new char[buffer_size_];
    client_sock_fd_ = sock_fd;
    image_cache_server_ = server;
}

shine::ImageConnection::~ImageConnection() {
    delete [] buffer_;
    buffer_ = nullptr;
    done_flag_ = false;
}

void shine::ImageConnection::run() {
    while (done_flag_) {
        // read data from sock_fd here
    }

}

bool shine::ImageConnection::done() {
    return done_flag_;
}