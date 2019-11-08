#include "ImageDataConnection.hh"
#include "ImageCache.hh"
#include "ImageData.hh"
#include "Decoder.hh"

#include <unistd.h>
#include <netinet/in.h>
#include <algorithm>
#include <boost/log/trivial.hpp>

using std::copy;

shine::ImageDataConnection::ImageDataConnection(
    int sock_fd, ImageCache* img_cache_):
    GeneralConnection(sock_fd, 100 * 1024 * 1024, 10 * 1024 * 1024, 0xBBBBCCCC) {
    image_cache_ = img_cache_;
}

shine::ImageDataConnection::~ImageDataConnection() {

}

void shine::ImageDataConnection::before_transferring_() {
    BOOST_LOG_TRIVIAL(info) << "connection ID: " << get_connection_id_();
}

bool shine::ImageDataConnection::do_transferring_() {
    // read image data request message
    slice_begin_ = 0;
    while (true) {
        const int slice_length = read(client_sock_fd_, buffer_ + slice_begin_, buffer_size_ - slice_begin_);
        if (slice_length == 0) {
            BOOST_LOG_TRIVIAL(warning) << "socket " << client_sock_fd_ << " is closed.";
            return false;
        }
        if (slice_begin_ + slice_length < 12) {
            slice_begin_ += slice_length;
        } else {
            break;
        }
    }

    uint32_t head = gDC.decode_byte<int32_t>(buffer_, 0, 3);
    uint32_t size = gDC.decode_byte<int32_t>(buffer_, 4, 7);
    if (head != 0xABABCAFE || size != 4) {
        BOOST_LOG_TRIVIAL(warning) << "got wrong image data request message, close the connection.";
        return false;
    }
    int img_count = gDC.decode_byte<int32_t>(buffer_, 8, 11);
    if (img_count < 1) {
        BOOST_LOG_TRIVIAL(warning) << "got invalid image count: " << img_count << ", close the connection.";
        return false;
    }
    if (img_count > 10) {
        BOOST_LOG_TRIVIAL(warning) << "got too large image count: " << img_count << ", close the connection.";
        return false;
    }
    BOOST_LOG_TRIVIAL(info) << "got one request with " << img_count << " images.";

    // send image data from here
    for (int i = 0; i < img_count; i++) {
        ImageData image_data;
        while (!image_cache_->take_one_image(image_data, 50)) {
            if (done_flag_) return false;
        }
        // serialize image_data into buffer
        // send data in buffer
    }


    return true;
}
