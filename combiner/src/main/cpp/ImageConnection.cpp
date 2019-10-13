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
}

void shine::ImageConnection::run() {
    if (start_connection()) {
        while (done_flag_) transfering();
    }
    close(client_sock_fd_);
    done_flag_ = true;
    return;
}

bool shine::ImageConnection::done() {
    return done_flag_;
}

bool shine::ImageConnection::start_connection() {
    int read_size = read(client_sock_fd_, buffer_, buffer_size_);

    return true;
}

void shine::ImageConnection::transfering() {
    int read_size = read(client_sock_fd_, buffer_, buffer_size_);
    if (read_size < 0) {
        cout << "socket is closed by the client." << endl;
        done_flag_ = false;
        return ;
    }

    // push data into image_cache_

}