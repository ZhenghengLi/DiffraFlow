#include "ImageConnection.hpp"
#include "ImageCache.hpp"

#include <unistd.h>
#include <netinet/in.h>
#include <algorithm>

using std::cout;
using std::cerr;
using std::endl;

shine::ImageConnection::ImageConnection(int sock_fd, ImageCache* img_cache_) {
    buffer_size_ = 1024 * 1024;
    buffer_ = new char[buffer_size_];
    slice_begin_ = 0;
    slice_length_ = buffer_size_;
    client_sock_fd_ = sock_fd;
    image_cache_ = img_cache_;
}

shine::ImageConnection::~ImageConnection() {
    delete [] buffer_;
}

void shine::ImageConnection::run() {
    if (start_connection_()) {
        while (done_flag_) transferring_();
    }
    close(client_sock_fd_);
    done_flag_ = true;
    return;
}

bool shine::ImageConnection::done() {
    return done_flag_;
}

bool shine::ImageConnection::start_connection_() {
    const int slice_length = read(client_sock_fd_, buffer_ + slice_begin_, buffer_size_ - slice_begin_);
    if (slice_length < 0) {
        cout << "socket is closed by the client." << endl;
        done_flag_ = false;
        return false; 
    }
    if (slice_begin_  + slice_length < 8) {
        slice_begin_ += slice_length;
        return true;
    }
    uint32_t success_code = htonl(200);
    uint32_t failure_code = htonl(300);
    int8_t header = decode_byte<int8_t>(buffer_, 0, 3);
    int8_t size = decode_byte<int8_t>(buffer_, 4, 7);
    if (header != 0xAAAABBBB || size != 0) {
        cout << "got wrong greeting message, close the connection." << endl;
        write(client_sock_fd_, &failure_code, 4);
        done_flag_ = false;
        return false;
    }
    write(client_sock_fd_, &success_code, 4);
    // ready for transferring data
    slice_begin_ = 0;
    return true;
}

void shine::ImageConnection::transferring_() {
    const int slice_length = read(client_sock_fd_, buffer_ + slice_begin_, buffer_size_ - slice_begin_);
    if (slice_length < 0) {
        cout << "socket is closed by the client." << endl;
        done_flag_ = false;
        return; 
    }

    // push data into image_cache_

}