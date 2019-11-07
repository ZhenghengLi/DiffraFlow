#include "ImageFrameConnection.hh"
#include "ImageCache.hh"
#include "ImageFrame.hh"
#include "Decoder.hh"

#include <unistd.h>
#include <netinet/in.h>
#include <algorithm>

using std::cout;
using std::cerr;
using std::endl;
using std::copy;

shine::ImageFrameConnection::ImageFrameConnection(
    int sock_fd, ImageCache* img_cache_):
    GeneralConnection(sock_fd, 100 * 1024 * 1024, 1024 * 1024, 0xAAAABBBB) {
    image_cache_ = img_cache_;
}

shine::ImageFrameConnection::~ImageFrameConnection() {

}

void shine::ImageFrameConnection::before_transferring_() {
    cout << "connection ID: " << get_connection_id_() << endl;
}

bool shine::ImageFrameConnection::do_transferring_() {
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
        uint32_t head = gDC.decode_byte<uint32_t>(buffer_ + position, 0, 3);
        position += 4;
        uint32_t size = gDC.decode_byte<uint32_t>(buffer_ + position, 0, 3);
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
