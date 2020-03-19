#include "CmbImgFrmConn.hh"
#include "CmbImgCache.hh"
#include "ImageFrame.hh"
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

log4cxx::LoggerPtr diffraflow::CmbImgFrmConn::logger_
    = log4cxx::Logger::getLogger("CmbImgFrmConn");

diffraflow::CmbImgFrmConn::CmbImgFrmConn(
    int sock_fd, CmbImgCache* img_cache_):
    GenericConnection(sock_fd, 0xDDCC1234, 0xDDD22CCC, 0xCCC22DDD, 100 * 1024 * 1024, 4 * 1024 * 1024) {
    image_cache_ = img_cache_;
    // note: make sure that this pkt_maxlen_ is larger than the send buffer of dispatcher
    buffer_uncompress_ = new char[pkt_maxlen_];
    buffer_uncompress_limit_ = 0;
}

diffraflow::CmbImgFrmConn::~CmbImgFrmConn() {
    delete [] buffer_uncompress_;
}

diffraflow::CmbImgFrmConn::ProcessRes diffraflow::CmbImgFrmConn::process_payload_(
    const char* payload_buffer, const uint32_t payload_size, const uint32_t payload_type) {

    // payload type check
    if (payload_type != 0xABCDFF00
        && payload_type != 0xABCDFF01
        && payload_type != 0xABCDFF02
        && payload_type != 0xABCDFF03) {
        LOG4CXX_INFO(logger_, "got unknown payload, do nothing and jump it.");
        return kContinue;
    }
    // payload size check
    if (payload_size < 4) {
        LOG4CXX_WARN(logger_, "got wrong image frame sequence data, close the connection.");
        return kStop;
    }

    // extract image counts
    uint32_t image_counts = gDC.decode_byte<uint32_t>(payload_buffer, 0, 3);
    if (image_counts == 0) {
        LOG4CXX_WARN(logger_, "got unexpected zero number_of_images, close the connection.");
        return kStop;

    }

    // - directly process without uncompressing
    const char* current_buffer = payload_buffer + 4;
    size_t current_limit = payload_size - 4;

    // - uncompress and process if the payload is compressed
    if (payload_type == 0xABCDFF01) {
        int result = LZ4_decompress_safe(payload_buffer + 4,
            buffer_uncompress_, payload_size - 4, pkt_maxlen_);
        if (result < 0) {
            LOG4CXX_WARN(logger_, "Failed to decompress data by LZ4 with error code: "
                << result << ", skip the packet.");
            return kContinue;
        } else {
            buffer_uncompress_limit_ = result;
        }
        current_buffer = buffer_uncompress_;
        current_limit  = buffer_uncompress_limit_;
    } else if (payload_type == 0xABCDFF02) {
        if (!snappy::GetUncompressedLength(payload_buffer + 4,
            payload_size - 4, &buffer_uncompress_limit_)) {
            LOG4CXX_WARN(logger_, "Failed to GetUncompressedLength, skip the packet.");
            return kContinue;
        }
        if (buffer_uncompress_limit_ > pkt_maxlen_) {
            LOG4CXX_WARN(logger_, "buffer_uncompress_limit_ > pkt_maxlen_, skip the packet.");
            return kContinue;
        }
        if (!snappy::RawUncompress(payload_buffer + 4,
            payload_size - 4, buffer_uncompress_)) {
            LOG4CXX_WARN(logger_, "Failed to RawUncompress, skip the packet.");
            return kContinue;
        }
        current_buffer = buffer_uncompress_;
        current_limit  = buffer_uncompress_limit_;
    } else if (payload_type == 0xABCDFF03) {
        buffer_uncompress_limit_ = ZSTD_decompress(buffer_uncompress_, pkt_maxlen_,
            payload_buffer + 4, payload_size - 4);
        if (ZSTD_isError(buffer_uncompress_limit_)) {
            LOG4CXX_WARN(logger_, "Failed to decompress data by ZSTD with error: "
                << ZSTD_getErrorName(buffer_uncompress_limit_) << ", skip the packet.");
            return kContinue;
        }
        current_buffer = buffer_uncompress_;
        current_limit  = buffer_uncompress_limit_;
    }

    LOG4CXX_DEBUG(logger_, "raw size = " << payload_size - 4 << ", processed size = " << current_limit);

    // process data in current_buffer
    size_t current_position = 0;
    for (size_t i = 0; i < image_counts; i++) {
        if (current_limit - current_position <= 4) {
            LOG4CXX_WARN(logger_, "unexpectedly reach the end of image frame sequence data, close the connection.");
            return kStop;
        }
        size_t current_size = gDC.decode_byte<uint32_t>(current_buffer + current_position, 0, 3);
        if (current_size == 0) {
            LOG4CXX_WARN(logger_, "got zero image frame size, close the connection.");
            return kStop;
        }
        current_position += 4;
        ImageFrame image_frame;
        image_frame.decode(current_buffer + current_position, current_size);
        current_position += current_size;
        image_cache_->push_frame(image_frame);
    }

    // size validation
    if (current_position != current_limit) {
        LOG4CXX_WARN(logger_, "got abnormal image frame sequence data, close the connection.");
        return kStop;
    }

    return kContinue;
}
