#include "CmbImgDataConn.hh"
#include "CmbImgCache.hh"
#include "ImageData.hh"
#include "Decoder.hh"

#include <unistd.h>
#include <netinet/in.h>
#include <algorithm>
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

using std::copy;

diffraflow::CmbImgDataConn::CmbImgDataConn(
    int sock_fd, CmbImgCache* img_cache_):
    GenericConnection(sock_fd, 0xBBBBCCCC, 0x0, 0x0, 100 * 1024 * 1024, 10 * 1024 * 1024) {
    image_cache_ = img_cache_;
    logger_ = log4cxx::Logger::getLogger("CmbImgDataConn");
}

diffraflow::CmbImgDataConn::~CmbImgDataConn() {
    log4cxx::NDC::remove();
}

void diffraflow::CmbImgDataConn::before_transferring_() {
    LOG4CXX_INFO(logger_, "connection ID: " << connection_id_);
}

bool diffraflow::CmbImgDataConn::do_transferring_() {
    // read image data request message
    slice_begin_ = 0;
    while (true) {
        const int slice_length = read(client_sock_fd_, buffer_ + slice_begin_, buffer_size_ - slice_begin_);
        if (slice_length == 0) {
            LOG4CXX_WARN(logger_, "socket " << client_sock_fd_ << " is closed.");
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
        LOG4CXX_WARN(logger_, "got wrong image data request message, close the connection.");
        return false;
    }
    int img_count = gDC.decode_byte<int32_t>(buffer_, 8, 11);
    if (img_count < 1) {
        LOG4CXX_WARN(logger_, "got invalid image count: " << img_count << ", close the connection.");
        return false;
    }
    if (img_count > 10) {
        LOG4CXX_WARN(logger_, "got too large image count: " << img_count << ", close the connection.");
        return false;
    }
    LOG4CXX_INFO(logger_, "got one request with " << img_count << " images.");

    // send image data from here
    for (int i = 0; i < img_count; i++) {
        ImageData image_data;
        while (!image_cache_->take_one_image(image_data)) {
            if (done_flag_) return false;
            if (image_cache_->img_queue_stopped()) return false;
        }
        // serialize image_data into buffer
        // send the data in buffer
        // check if the sock_fd is closed when writting data
        // if closed return false;
        // note: like read(), when write() return 0, it means the sock_fd is closed.

    }

    return true;
}
