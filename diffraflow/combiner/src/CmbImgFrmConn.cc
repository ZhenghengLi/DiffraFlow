#include "CmbImgFrmConn.hh"
#include "CmbImgCache.hh"
#include "ImageFrameRaw.hh"
#include "Decoder.hh"

#include <unistd.h>
#include <netinet/in.h>
#include <algorithm>
#include <string>

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <snappy.h>
#include <lz4.h>
#include <zstd.h>

using std::copy;

log4cxx::LoggerPtr diffraflow::CmbImgFrmConn::logger_ = log4cxx::Logger::getLogger("CmbImgFrmConn");

diffraflow::CmbImgFrmConn::CmbImgFrmConn(int sock_fd, CmbImgCache* img_cache_)
    : GenericConnection(sock_fd, 0xDDCC1234, 0xDDD22CCC, 0xCCC22DDD, 6 * 1024 * 1024) {
    image_cache_ = img_cache_;
}

diffraflow::CmbImgFrmConn::~CmbImgFrmConn() {}

bool diffraflow::CmbImgFrmConn::do_receiving_and_processing_() {
    uint32_t payload_type = 0;

    shared_ptr<ByteBuffer> payload_data;
    if (!receive_one_(payload_type, payload_data)) {
        return false;
    }

    switch (payload_type) {
    case 0xABCDFFFF: {
        if (payload_data->size() != 131096) {
            LOG4CXX_INFO(logger_, "got an image frame with wrong size " << payload_data->size() << ", skip it.");
            return true;
        }
        shared_ptr<ImageFrameRaw> image_frame = make_shared<ImageFrameRaw>();
        if (!image_frame->set_data(payload_data)) {
            LOG4CXX_WARN(logger_, "failed to set image frame, skip it.");
            return true;
        }
        if (image_cache_->push_frame(image_frame)) {
            LOG4CXX_DEBUG(logger_, "successfully pushed image frame into image cache.");
        } else {
            LOG4CXX_WARN(logger_, "failed to push image frame into image cache, skip it.");
            return true;
        }
    } break;
    default:
        LOG4CXX_INFO(logger_, "got unknown payload, do nothing and jump it.")
    }

    return true;
}
