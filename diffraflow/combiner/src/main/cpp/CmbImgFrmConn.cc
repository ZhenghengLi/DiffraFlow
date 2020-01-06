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
    GenericConnection(sock_fd, 0xDDCC1234, 0xDDD22CCC, 0xCCC22DDD, 100 * 1024 * 1024, 1024 * 1024) {
    image_cache_ = img_cache_;
}

diffraflow::CmbImgFrmConn::~CmbImgFrmConn() {

}

diffraflow::CmbImgFrmConn::ProcessRes diffraflow::CmbImgFrmConn::process_payload_(
    const size_t payload_position, const uint32_t payload_size, const uint32_t payload_type) {
    char* current_buffer = buffer_ + payload_position;
    if (payload_type == 0xABCDF8F8) {
        // uncompressed image frame sequence data
        if (payload_size < 4) {
            BOOST_LOG_TRIVIAL(warning) << "got wrong image frame sequence data, close the connection.";
            return kStop;
        }
        uint32_t number_of_images = gDC.decode_byte<uint32_t>(current_buffer, 0, 3);
        if (number_of_images == 0) {
            BOOST_LOG_TRIVIAL(warning) << "got zero number_of_images, close the connection.";
            return kStop;

        }
        current_buffer += 4;
        for (size_t i = 0; i < number_of_images; i++) {
            if (current_buffer - buffer_ - payload_position >= payload_size - 4) {
                BOOST_LOG_TRIVIAL(warning) << "unexpectedly reach the end of image frame sequence data, close the connection.";
                return kStop;
            }
            size_t current_size = gDC.decode_byte<uint32_t>(current_buffer, 0, 3);
            if (current_size == 0) {
                BOOST_LOG_TRIVIAL(warning) << "got zero image frame size, close the connection.";
                return kStop;
            }
            current_buffer += 4;
            ImageFrame image_frame;
            image_frame.decode(current_buffer, current_size);
            current_buffer += current_size;
            image_cache_->put_frame(image_frame);
        }
        if (current_buffer - buffer_ != payload_position + payload_size) {
            BOOST_LOG_TRIVIAL(warning) << "got abnormal image frame sequence data, close the connection.";
            return kStop;
        }
    } else {
        BOOST_LOG_TRIVIAL(info) << "got unknown payload, do nothing and jump it.";
    }
    return kContinue;
}
