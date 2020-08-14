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
    // note: make sure that this pkt_maxlen_ is larger than the send buffer of dispatcher
    buffer_uncompress_ = new char[buffer_size_];
    buffer_uncompress_limit_ = 0;

    frame_metrics.total_processed_frame_size = 0;
    frame_metrics.total_processed_frame_counts = 0;

    compression_metrics.total_compressed_size = 0;
    compression_metrics.total_uncompressed_size = 0;
}

diffraflow::CmbImgFrmConn::~CmbImgFrmConn() { delete[] buffer_uncompress_; }

diffraflow::GenericConnection::ProcessRes diffraflow::CmbImgFrmConn::process_payload_(
    const char* payload_buffer, const size_t payload_size) {

    uint32_t payload_type = gDC.decode_byte<uint32_t>(payload_buffer, 0, 3);
    switch (payload_type) {
    case 0xABCDFFFF: {
        const char* frame_buffer = payload_buffer + 4;
        size_t frame_size = payload_size - 4;
        if (frame_size != 131096) {
            LOG4CXX_INFO(logger_, "got an image frame with wrong size " << frame_size << ", skip it.");
            return kSkipped;
        }
        shared_ptr<ImageFrameRaw> image_frame = make_shared<ImageFrameRaw>();
        if (!image_frame->set_data(frame_buffer, frame_size)) {
            LOG4CXX_WARN(logger_, "failed to copy image frame, skip it.");
            return kSkipped;
        }
        if (image_cache_->push_frame(image_frame)) {
            LOG4CXX_DEBUG(logger_, "successfully pushed image frame into image cache.");
        } else {
            LOG4CXX_WARN(logger_, "failed to push image frame into image cache, skip it.");
            return kSkipped;
        }
    } break;
    default:
        LOG4CXX_INFO(logger_, "got unknown payload, do nothing and jump it.");
        return kSkipped;
    }

    return kProcessed;
}

json::value diffraflow::CmbImgFrmConn::collect_metrics() {

    json::value root_json = GenericConnection::collect_metrics();

    json::value frame_metrics_json;
    frame_metrics_json["total_processed_frame_size"] =
        json::value::number(frame_metrics.total_processed_frame_size.load());
    frame_metrics_json["total_processed_frame_counts"] =
        json::value::number(frame_metrics.total_processed_frame_counts.load());

    json::value compression_metrics_json;
    compression_metrics_json["total_compressed_size"] =
        json::value::number(compression_metrics.total_compressed_size.load());
    compression_metrics_json["total_uncompressed_size"] =
        json::value::number(compression_metrics.total_uncompressed_size.load());

    root_json["frame_stats"] = frame_metrics_json;
    root_json["compression_stats"] = compression_metrics_json;

    return root_json;
}