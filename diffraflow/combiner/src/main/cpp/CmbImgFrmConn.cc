#include "CmbImgFrmConn.hh"
#include "CmbImgCache.hh"
#include "ImageFrame.hh"
#include "Decoder.hh"

#include <unistd.h>
#include <netinet/in.h>
#include <algorithm>
#include <boost/log/trivial.hpp>

using std::copy;

diffraflow::CmbImgFrmConn::CmbImgFrmConn(
    int sock_fd, CmbImgCache* img_cache_):
    GenericConnection(sock_fd, 100 * 1024 * 1024, 1024 * 1024, 0xAAAABBBB) {
    image_cache_ = img_cache_;
}

diffraflow::CmbImgFrmConn::~CmbImgFrmConn() {

}

void diffraflow::CmbImgFrmConn::before_transferring_() {
    BOOST_LOG_TRIVIAL(info) << "connection ID: " << get_connection_id_();
}

bool diffraflow::CmbImgFrmConn::do_transferring_() {
    const int slice_length = read(client_sock_fd_, buffer_ + slice_begin_, buffer_size_ - slice_begin_);
    if (slice_length == 0) {
        BOOST_LOG_TRIVIAL(info) << "socket " << client_sock_fd_ << " is closed.";
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
        uint32_t head = gDC.decode_byte<uint32_t>(buffer_ + position, 0, 3);
        position += 4;
        uint32_t size = gDC.decode_byte<uint32_t>(buffer_ + position, 0, 3);
        position += 4;

        // validation check for packet
        if (size > pkt_maxlen_) {
            BOOST_LOG_TRIVIAL(info) << "got too large packet, close the connection.";
            return false;
        }
        if (head == 0xABCDEEFF && size <= 8) {
            BOOST_LOG_TRIVIAL(info) << "got wrong image packet, close the connection.";
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
            BOOST_LOG_TRIVIAL(info) << "got unknown packet, jump it.";
            position += size;
        }

    }
    return true;
}
