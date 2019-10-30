#include "ImageConnection.hpp"
#include "ImageCache.hpp"
#include "ImageFrame.hpp"

#include <unistd.h>
#include <netinet/in.h>
#include <algorithm>

using std::cout;
using std::cerr;
using std::endl;
using std::copy;

shine::ImageConnection::ImageConnection(int sock_fd, ImageCache* img_cache_) {
    buffer_size_ = 100 * 1024 * 1024;
    buffer_ = new char[buffer_size_];
    slice_begin_ = 0;
    pkt_maxlen_ = 1024 * 1024;
    pkt_data_ = new char[pkt_maxlen_];
    client_sock_fd_ = sock_fd;
    image_cache_ = img_cache_;
    done_flag_ = false;
}

shine::ImageConnection::~ImageConnection() {
    delete [] buffer_;
}

void shine::ImageConnection::run() {
    if (start_connection_()) {
        while (!done_flag_ && transferring_());
    }
    close(client_sock_fd_);
    done_flag_ = true;
    return;
}

bool shine::ImageConnection::done() {
    return done_flag_;
}

void shine::ImageConnection::shift_left_(const size_t position, const size_t limit) {
    if (position == 0) {
        copy(buffer_ + position, buffer_ + limit, buffer_);
    }
    slice_begin_ = limit - position;
}

bool shine::ImageConnection::start_connection_() {
    while (true) {
        const int slice_length = read(client_sock_fd_, buffer_ + slice_begin_, buffer_size_ - slice_begin_);
        if (slice_length == 0) {
            cout << "socket is closed by the client." << endl;
            return false; 
        }
        if (slice_begin_ + slice_length < 12) {
            slice_begin_ += slice_length;
        } else {
            break;
        }
    }
    uint32_t success_code = htonl(200);
    uint32_t failure_code = htonl(300);
    int32_t head = decode_byte<int32_t>(buffer_, 0, 3);
    int32_t size = decode_byte<int32_t>(buffer_, 4, 7);
    if (head != 0xAAAABBBB || size != 4) {
        cout << "got wrong greeting message, close the connection." << endl;
        write(client_sock_fd_, &failure_code, 4);
        done_flag_ = false;
        return false;
    }
    int32_t conn_id = decode_byte<int32_t>(buffer_, 8, 11);
    cout << "connection ID: " << conn_id << endl;
    write(client_sock_fd_, &success_code, 4);
    // ready for transferring data
    slice_begin_ = 0;
    return true;
}

bool shine::ImageConnection::transferring_() {
    const int slice_length = read(client_sock_fd_, buffer_ + slice_begin_, buffer_size_ - slice_begin_);
    if (slice_length == 0) {
        cout << "socket is closed by the client." << endl;
        return false; 
    }
    const size_t limit = slice_begin_ + slice_length;
    size_t position = 0;
    while (true) {
        if (limit - position < 8) {
            shift_left_(position, limit);
            break;
        }

        // read head and size
        uint32_t head = decode_byte<uint32_t>(buffer_ + position, 0, 3);
        position += 4;
        uint32_t size = decode_byte<uint32_t>(buffer_ + position, 0, 3);
        position += 4;

        // validation check for packet
        if (size > pkt_maxlen_) {
            cout << "got too large packet, close the connection." << endl;
            return false;
        }
        if (head == 0xABCDEEFF && size <= 8) {
            cout << "got wrong image packet, close the connection." << endl;
            return false;
        }

        // continue to receive more data
        if (limit - position < size) {
            position -= 8;
            shift_left_(position, limit);
            break;
        }

        // decode one packet

        // 0xABCDEEFF
        ImageFrame image_frame;

        switch (head) {
        case 0xABCDEEFF: // image data
            image_frame.decode(buffer_ + position, size);
            image_cache_->put_frame(image_frame);
            position += size;
            break;
        default:
            cout << "got unknown packet, jump it." << endl;
            position += size;
        }

    }
    return true;
}
