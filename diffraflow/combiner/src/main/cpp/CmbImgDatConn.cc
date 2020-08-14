#include "CmbImgDatConn.hh"
#include "CmbImgCache.hh"
#include "ImageDataRaw.hh"
#include "ImageFrameRaw.hh"
#include "Decoder.hh"
#include "PrimitiveSerializer.hh"

#include <unistd.h>
#include <netinet/in.h>
#include <algorithm>
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

using std::copy;
using std::shared_ptr;

log4cxx::LoggerPtr diffraflow::CmbImgDatConn::logger_ = log4cxx::Logger::getLogger("CmbImgDatConn");

diffraflow::CmbImgDatConn::CmbImgDatConn(int sock_fd, CmbImgCache* image_cache, size_t max_req_imgct)
    : GenericConnection(sock_fd, 0xEECC1234, 0xEEE22CCC, 0xCCC22EEE, 1024) {
    image_cache_ = image_cache;
    max_req_imgct_ = max_req_imgct;

    image_metrics.total_sent_images = 0;
}

diffraflow::CmbImgDatConn::~CmbImgDatConn() {}

bool diffraflow::CmbImgDatConn::do_preparing_and_sending_() {
    // serialize and send image data
    shared_ptr<ImageDataRaw> one_image;
    if (!image_cache_->take_image(one_image)) {
        LOG4CXX_WARN(logger_, "image data queue is stopped and empty, close the connection.");
        return false;
    }
    // serialize image data
    // image_buffer_.clear();
    // msgpack::pack(image_buffer_, *one_image);

    return true;

    // TODO: send one_image without data copy

    char meta_buffer[11];
    one_image->serialize_meta(meta_buffer, 11);

    // (1) calcaulte size
    uint32_t image_size = 11;
    for (size_t i = 0; i < one_image->alignment_vec.size(); i++) {
        if (one_image->alignment_vec[i]) {
            image_size += one_image->image_frame_vec[i]->size();
        }
    }
    // (2) send head and size => send_head_(uint32_t size);
    // send_head_(image_size);

    // (3) send meta-data of one_image => send_segment_(const char* buffer, size_t len)
    // send_segment_(meta_buffer, 11);

    // (4) send each image_frame one by one
    for (size_t i = 0; i < one_image->alignment_vec.size(); i++) {
        if (one_image->alignment_vec[i]) {
            // send_segment_(one_image->image_frame_vec[i]->data(), one_image->image_frame_vec[i]->size());
        }
    }

    // the following code is not needed.

    // serialize head
    char head_buffer[4];
    gPS.serializeValue<uint32_t>(0xABCDEEEE, head_buffer, 4);
    // send data
    if (send_one_(head_buffer, 4, image_buffer_.data(), image_buffer_.size())) {
        LOG4CXX_DEBUG(logger_, "successfully send one image.");
        image_metrics.total_sent_images++;
        return true;
    } else {
        LOG4CXX_ERROR(logger_, "failed to send one image.");
        return false;
    }
}

json::value diffraflow::CmbImgDatConn::collect_metrics() {
    json::value root_json = GenericConnection::collect_metrics();
    json::value image_metrics_json;
    image_metrics_json["total_sent_images"] = json::value::number(image_metrics.total_sent_images.load());
    root_json["image_stats"] = image_metrics_json;
    return root_json;
}
