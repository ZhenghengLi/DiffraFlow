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
    if (payload_type == 0xABCDFFFF && payload_size <= 8) {
        BOOST_LOG_TRIVIAL(info) << "got wrong image frame, close the connection.";
        return kStop;
    }
    // decode one image frame
    if (payload_type == 0xABCDFFFF) {
        ImageFrame image_frame;
        image_frame.decode(buffer_ + payload_position, payload_size);
        image_cache_->put_frame(image_frame);
    } else {
        BOOST_LOG_TRIVIAL(info) << "got unknown payload, do nothing and jump it.";
    }
    return kContinue;
}
