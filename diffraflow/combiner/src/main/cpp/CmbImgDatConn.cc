#include "CmbImgDatConn.hh"
#include "CmbImgCache.hh"
#include "ImageData.hh"
#include "Decoder.hh"
#include "PrimitiveSerializer.hh"

#include <unistd.h>
#include <netinet/in.h>
#include <algorithm>
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

using std::copy;

log4cxx::LoggerPtr diffraflow::CmbImgDatConn::logger_
    = log4cxx::Logger::getLogger("CmbImgDatConn");

diffraflow::CmbImgDatConn::CmbImgDatConn(
    int sock_fd, CmbImgCache* image_cache, size_t max_req_imgct):
    GenericConnection(sock_fd, 0xEECC1234, 0xEEE22CCC, 0xCCC22EEE, 1024) {
    image_cache_ = image_cache;
    max_req_imgct_ = max_req_imgct;
}

diffraflow::CmbImgDatConn::~CmbImgDatConn() {

}

diffraflow::GenericConnection::ProcessRes diffraflow::CmbImgDatConn::process_payload_(
    const char* payload_buffer, const size_t payload_size) {
    // payload size check
    if (payload_size < 8) {
        LOG4CXX_WARN(logger_, "got too short image request data, close the connection.");
        return kFailed;
    }
    // extract payload type
    uint32_t payload_type = gDC.decode_byte<uint32_t>(payload_buffer, 0, 3);
    if (payload_type != 0xEEEEABCD) {
        LOG4CXX_WARN(logger_, "got unknown payload type, close the connection.");
        return kFailed;
    }
    // extract number of images
    uint32_t image_counts = gDC.decode_byte<uint32_t>(payload_buffer, 4, 7);
    if (image_counts < 1 || image_counts > max_req_imgct_) {
        LOG4CXX_WARN(logger_, "image counts is out of range " << 1 << "-" << max_req_imgct_ << ", close the connection.");
        return kFailed;
    }

    // serialize and send image data
    for (size_t i = 0; i < image_counts; i++) {
        ImageData one_image;
        if (!image_cache_->take_image(one_image)) {
            LOG4CXX_WARN(logger_, "image data queue is stopped and empty, close the connection.");
            return kFailed;
        }
        // serialize image data
        image_buffer_.clear();
        msgpack::pack(image_buffer_, one_image);
        // serialize head
        char head_buffer[4];
        gPS.serializeValue<uint32_t>(0xABCDEEEE, head_buffer, 4);
        // send data
        if (send_one_(head_buffer, 4, image_buffer_.data(), image_buffer_.size())) {
            LOG4CXX_DEBUG(logger_, "successfully send one image.");
        } else {
            LOG4CXX_ERROR(logger_, "failed to send one image.");
            return kFailed;
        }
    }

    return ProcessRes::kProcessed;
}