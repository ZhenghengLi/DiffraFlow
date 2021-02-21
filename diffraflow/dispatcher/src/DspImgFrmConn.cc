#include "DspImgFrmConn.hh"
#include "PrimitiveSerializer.hh"
#include "Decoder.hh"
#include "DspSender.hh"
#include "ImageFrameRaw.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

log4cxx::LoggerPtr diffraflow::DspImgFrmConn::logger_ = log4cxx::Logger::getLogger("DspImgFrmConn");

diffraflow::DspImgFrmConn::DspImgFrmConn(int sock_fd, DspSender** sender_arr, size_t sender_cnt)
    : GenericConnection(sock_fd, 0xFFDD1234, 0xFFF22DDD, 0xDDD22FFF, 1024 * 1024) {
    sender_array_ = sender_arr;
    sender_count_ = sender_cnt;
}

diffraflow::DspImgFrmConn::~DspImgFrmConn() {}

bool diffraflow::DspImgFrmConn::do_receiving_and_processing_() {

    uint32_t payload_type = 0;
    shared_ptr<vector<char>> payload_data;
    if (!receive_one_(payload_type, payload_data)) {
        return false;
    }

    size_t index = 0;
    shared_ptr<ImageFrameRaw> image_frame = make_shared<ImageFrameRaw>();

    switch (payload_type) {
    case 0xABCDFFFF:
        if (payload_data->size() != 131096) {
            LOG4CXX_INFO(logger_, "got an image frame with wrong size " << payload_data->size() << ", skip it.");
            return true;
        }
        if (!image_frame->set_data(payload_data)) {
            LOG4CXX_WARN(logger_, "failed to set image frame, skip it.");
            return true;
        }

        index = hash_long_(image_frame->get_key()) % sender_count_;
        if (sender_array_[index]->push(image_frame)) {
            LOG4CXX_DEBUG(logger_, "push one image frame into sender[" << index << "].");
            return true;
        } else {
            LOG4CXX_WARN(logger_, "sender[" << index << "] is stopped, close the connection.");
            return false;
        }
        break;
    default:
        LOG4CXX_INFO(logger_, "got unknown payload, do nothing and jump it.")
        return true;
    }
}

uint32_t diffraflow::DspImgFrmConn::hash_long_(uint64_t value) { return (uint32_t)(value ^ (value >> 32)); }
