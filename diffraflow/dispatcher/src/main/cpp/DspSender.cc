#include "DspSender.hh"
#include "PrimitiveSerializer.hh"
#include "ImageFrameRaw.hh"

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

log4cxx::LoggerPtr diffraflow::DspSender::logger_ = log4cxx::Logger::getLogger("DspSender");

diffraflow::DspSender::DspSender(string hostname, int port, int id, size_t max_qs)
    : GenericClient(hostname, port, id, 0xDDCC1234, 0xDDD22CCC, 0xCCC22DDD) {
    sending_thread_ = nullptr;
    imgfrm_queue_.set_maxsize(max_qs);

    sender_metrics.total_pushed_counts = 0;
    sender_metrics.total_send_succ_counts = 0;
    sender_metrics.total_send_fail_counts = 0;
}

diffraflow::DspSender::~DspSender() {}

bool diffraflow::DspSender::send_imgfrm_(const shared_ptr<ImageFrameRaw>& image_frame) {
    // try to connect if lose connection
    if (not_connected()) {
        if (connect_to_server()) {
            LOG4CXX_INFO(logger_, "reconnected to combiner.");
        } else {
            LOG4CXX_WARN(logger_, "failed to reconnect to combiner, discard data in buffer.");
            return false;
        }
    }
    char payload_type_buffer[4];
    gPS.serializeValue<uint32_t>(0xABCDFFFF, payload_type_buffer, 4);
    if (image_frame->data()) {
        if (send_one_(payload_type_buffer, 4, image_frame->data(), image_frame->size())) {
            return true;
        } else {
            close_connection();
            return false;
        }
    } else if (image_frame->get_dgram_count() > 0) {
        // (1) calculate size
        size_t frame_size = 4;
        for (size_t i = 0; i < image_frame->get_dgram_count(); i++) {
            frame_size += image_frame->get_dgram(i)->size() - 4;
        }
        // (2) send head and size
        if (!send_head_(frame_size)) {
            LOG4CXX_ERROR(logger_, "failed to send head.");
            close_connection();
            return false;
        }
        // (3) send payload type
        if (!send_segment_(payload_type_buffer, 4)) {
            LOG4CXX_ERROR(logger_, "failed to send payload type.");
            close_connection();
            return false;
        }
        // (4) send each frame segment
        for (size_t i = 0; i < image_frame->get_dgram_count(); i++) {
            if (!send_segment_(image_frame->get_dgram(i)->data() + 4, image_frame->get_dgram(i)->size() - 4)) {
                LOG4CXX_ERROR(logger_, "failed to send frame segment of index " << i << ".");
                close_connection();
                return false;
            }
        }
        return true;
    } else {
        LOG4CXX_ERROR(logger_, "cannot send empty image frame.");
        return false;
    }
}

bool diffraflow::DspSender::push(const shared_ptr<ImageFrameRaw>& image_frame) {
    bool result = imgfrm_queue_.push(image_frame);

    sender_metrics.total_pushed_counts++;

    return result;
}

bool diffraflow::DspSender::start() {
    if (sending_thread_ != nullptr) {
        LOG4CXX_WARN(logger_, "sender is already started with connection to combiner " << get_server_address() << ".");
        return true;
    }
    if (connect_to_server()) {
        LOG4CXX_INFO(logger_, "Successfully connected to combiner " << get_server_address() << ".");
    } else {
        LOG4CXX_ERROR(logger_, "Failed to connect to combiner " << get_server_address() << ".");
        return false;
    }
    sending_thread_ = new thread([this]() {
        shared_ptr<ImageFrameRaw> image_frame;
        while (imgfrm_queue_.take(image_frame)) {
            if (send_imgfrm_(image_frame)) {
                LOG4CXX_DEBUG(logger_, "successfully send one image frame.");
                sender_metrics.total_send_succ_counts++;
            } else {
                LOG4CXX_DEBUG(logger_, "failed to send one image frame.");
                sender_metrics.total_send_fail_counts++;
            }
        }
    });
    return true;
}

void diffraflow::DspSender::stop() {
    if (sending_thread_ == nullptr) return;
    imgfrm_queue_.stop();
    sending_thread_->join();
    delete sending_thread_;
    sending_thread_ = nullptr;
    close_connection();
}

json::value diffraflow::DspSender::collect_metrics() {

    json::value root_json = GenericClient::collect_metrics();

    json::value sender_metrics_json;
    sender_metrics_json["total_pushed_counts"] = json::value::number(sender_metrics.total_pushed_counts);
    sender_metrics_json["total_send_succ_counts"] = json::value::number(sender_metrics.total_send_succ_counts);
    sender_metrics_json["total_send_fail_counts"] = json::value::number(sender_metrics.total_send_fail_counts);
    sender_metrics_json["current_queue_size"] = json::value::number((uint64_t)imgfrm_queue_.size());

    root_json["sender_stats"] = sender_metrics_json;

    return root_json;
}