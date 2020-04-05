#include "CmbImgFrmConn.hh"
#include "CmbImgCache.hh"
#include "ImageFramePtr.hh"
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
    GenericConnection(sock_fd, 0xDDCC1234, 0xDDD22CCC, 0xCCC22DDD, 4 * 1024 * 1024) {
    image_cache_ = img_cache_;
    // note: make sure that this pkt_maxlen_ is larger than the send buffer of dispatcher
    buffer_uncompress_ = new char[buffer_size_];
    buffer_uncompress_limit_ = 0;

    frame_metrics.total_processed_frame_size = 0;
    frame_metrics.total_processed_frame_counts = 0;

    compression_metrics.total_compressed_size = 0;
    compression_metrics.total_uncompressed_size = 0;

}

diffraflow::CmbImgFrmConn::~CmbImgFrmConn() {
    delete [] buffer_uncompress_;
}

diffraflow::GenericConnection::ProcessRes diffraflow::CmbImgFrmConn::process_payload_(
    const char* payload_buffer, const size_t payload_size) {

    // payload size check
    if (payload_size < 8) {
        LOG4CXX_WARN(logger_, "got too short image frame sequence data, close the connection.");
        return kFailed;
    }
    // extract payload type
    uint32_t payload_type = gDC.decode_byte<uint32_t>(payload_buffer, 0, 3);
    // extract image counts
    uint32_t image_counts = gDC.decode_byte<uint32_t>(payload_buffer, 4, 7);
    if (image_counts == 0) {
        LOG4CXX_WARN(logger_, "got unexpected zero number_of_images, close the connection.");
        return kFailed;
    }

    const char* current_buffer = nullptr;
    size_t current_limit = 0;

    switch (payload_type) {
    case 0xABCDFF00: // Non-Compress, use the raw data
        {
            current_buffer = payload_buffer + 8;
            current_limit = payload_size - 8;
        }
        break;
    case 0xABCDFF01: // Compressed by LZ4, decompress
        {
            int result = LZ4_decompress_safe(payload_buffer + 8,
                buffer_uncompress_, payload_size - 8, buffer_size_);
            if (result < 0) {
                LOG4CXX_WARN(logger_, "Failed to decompress data by LZ4 with error code: "
                    << result << ", skip the packet.");
                return kSkipped;
            } else {
                buffer_uncompress_limit_ = result;
            }
            current_buffer = buffer_uncompress_;
            current_limit  = buffer_uncompress_limit_;
        }
        break;
    case 0xABCDFF02: // Compressed by Snappy, decompress
        {
            if (!snappy::GetUncompressedLength(payload_buffer + 8,
                payload_size - 8, &buffer_uncompress_limit_)) {
                LOG4CXX_WARN(logger_, "Failed to GetUncompressedLength, skip the packet.");
                return kSkipped;
            }
            if (buffer_uncompress_limit_ > buffer_size_) {
                LOG4CXX_WARN(logger_, "buffer_uncompress_limit_ > buffer_size_, skip the packet.");
                return kSkipped;
            }
            if (!snappy::RawUncompress(payload_buffer + 8,
                payload_size - 8, buffer_uncompress_)) {
                LOG4CXX_WARN(logger_, "Failed to RawUncompress, skip the packet.");
                return kSkipped;
            }
            current_buffer = buffer_uncompress_;
            current_limit  = buffer_uncompress_limit_;
        }
        break;
    case 0xABCDFF03: // Compressed by ZSTD, decompress
        {
            buffer_uncompress_limit_ = ZSTD_decompress(buffer_uncompress_, buffer_size_,
                payload_buffer + 8, payload_size - 8);
            if (ZSTD_isError(buffer_uncompress_limit_)) {
                LOG4CXX_WARN(logger_, "Failed to decompress data by ZSTD with error: "
                    << ZSTD_getErrorName(buffer_uncompress_limit_) << ", skip the packet.");
                return kSkipped;
            }
            current_buffer = buffer_uncompress_;
            current_limit  = buffer_uncompress_limit_;
        }
        break;
    default:
        LOG4CXX_INFO(logger_, "got unknown payload, do nothing and jump it.");
        return kSkipped;
    }

    LOG4CXX_DEBUG(logger_, "raw size = " << payload_size - 8 << ", processed size = " << current_limit);

    compression_metrics.total_compressed_size += payload_size - 8;
    compression_metrics.total_uncompressed_size += current_limit;

    // process data in current_buffer
    size_t current_position = 0;
    for (size_t i = 0; i < image_counts; i++) {
        if (current_limit - current_position <= 4) {
            LOG4CXX_WARN(logger_, "unexpectedly reach the end of image frame sequence data, close the connection.");
            return kFailed;
        }
        size_t current_size = gDC.decode_byte<uint32_t>(current_buffer + current_position, 0, 3);
        if (current_size == 0) {
            LOG4CXX_WARN(logger_, "got zero image frame size, close the connection.");
            return kFailed;
        }
        current_position += 4;
        ImageFramePtr image_frame(new ImageFrame());
        image_frame->decode(current_buffer + current_position, current_size);
        current_position += current_size;

        if (image_cache_->push_frame(image_frame)) {
            LOG4CXX_DEBUG(logger_, "successfully pushed image frame into image cache.");
        } else {
            LOG4CXX_WARN(logger_, "image cache is stopped, close the connection.");
            return kFailed;
        }

        frame_metrics.total_processed_frame_size += current_size;
        frame_metrics.total_processed_frame_counts += 1;

    }

    // size validation
    if (current_position != current_limit) {
        LOG4CXX_WARN(logger_, "got abnormal image frame sequence data, close the connection.");
        return kFailed;
    }

    return kProcessed;
}

json::value diffraflow::CmbImgFrmConn::collect_metrics() {

    json::value root_json = GenericConnection::collect_metrics();

    json::value frame_metrics_json;
    frame_metrics_json["total_processed_frame_size"] = json::value::number(frame_metrics.total_processed_frame_size.load());
    frame_metrics_json["total_processed_frame_counts"] = json::value::number(frame_metrics.total_processed_frame_counts.load());

    json::value compression_metrics_json;
    compression_metrics_json["total_compressed_size"] = json::value::number(compression_metrics.total_compressed_size.load());
    compression_metrics_json["total_uncompressed_size"] = json::value::number(compression_metrics.total_uncompressed_size.load());

    root_json["frame_stats"] = frame_metrics_json;
    root_json["compression_stats"] = compression_metrics_json;

    return root_json;

}