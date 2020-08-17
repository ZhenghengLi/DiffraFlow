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
    // take one_image from queue

    LOG4CXX_DEBUG(logger_, "debug: start of do_preparing_and_sender_().");

    shared_ptr<ImageDataRaw> one_image;
    if (!image_cache_->take_image(one_image)) {
        LOG4CXX_WARN(logger_, "image data queue is stopped and empty, close the connection.");
        return false;
    }

    LOG4CXX_DEBUG(logger_, "debug: after take_image(one_image).");
    one_image->print();

    return true;

    //    // send one_image without data copy
    //    // (1) serialize meta-data of one_image
    //    char meta_buffer[15];
    //    // - payload head
    //    gPS.serializeValue<uint32_t>(0xABCDEEEE, meta_buffer, 4);
    //    // - met-data
    //    one_image->serialize_meta(meta_buffer + 4, 11);

    //    LOG4CXX_DEBUG(logger_, "debug: after serialize_meta.");

    //    // (2) calcaulte size
    //    uint32_t image_size = 15;
    //    for (size_t i = 0; i < one_image->alignment_vec.size(); i++) {
    //        if (one_image->alignment_vec[i]) {
    //            image_size += one_image->image_frame_vec[i]->size();
    //        }
    //    }
    //    // (3) send head and size
    //    if (!send_head_(image_size)) {
    //        LOG4CXX_ERROR(logger_, "failed to send head.");
    //        return false;
    //    }
    //    // (4) send meta-data of one_image
    //    if (!send_segment_(meta_buffer, 15)) {
    //        LOG4CXX_ERROR(logger_, "failed to send meta data of image");
    //        return false;
    //    }
    //    // (5) send each image_frame one by one
    //    for (size_t i = 0; i < one_image->alignment_vec.size(); i++) {
    //        if (one_image->alignment_vec[i]) {
    //            if (!send_segment_(one_image->image_frame_vec[i]->data(), one_image->image_frame_vec[i]->size())) {
    //                LOG4CXX_ERROR(logger_, "failed to send image frame of module " << i << ".");
    //                return false;
    //            }
    //        }
    //    }

    //    LOG4CXX_DEBUG(logger_, "successfully send one image.");
    //    image_metrics.total_sent_images++;

    //    return true;
}

json::value diffraflow::CmbImgDatConn::collect_metrics() {
    json::value root_json = GenericConnection::collect_metrics();
    json::value image_metrics_json;
    image_metrics_json["total_sent_images"] = json::value::number(image_metrics.total_sent_images.load());
    root_json["image_stats"] = image_metrics_json;
    return root_json;
}
