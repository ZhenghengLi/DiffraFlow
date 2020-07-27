#include "DspImgFrmConn.hh"
#include "PrimitiveSerializer.hh"
#include "Decoder.hh"
#include "DspSender.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

log4cxx::LoggerPtr diffraflow::DspImgFrmConn::logger_ = log4cxx::Logger::getLogger("DspImgFrmConn");

diffraflow::DspImgFrmConn::DspImgFrmConn(int sock_fd, DspSender** sender_arr, size_t sender_cnt)
    : GenericConnection(sock_fd, 0xFFDD1234, 0xFFF22DDD, 0xDDD22FFF, 1024 * 1024) {
    sender_array_ = sender_arr;
    sender_count_ = sender_cnt;
}

diffraflow::DspImgFrmConn::~DspImgFrmConn() {}

diffraflow::GenericConnection::ProcessRes diffraflow::DspImgFrmConn::process_payload_(
    const char* payload_buffer, const size_t payload_size) {
    // extract payload type
    uint32_t payload_type = gDC.decode_byte<uint32_t>(payload_buffer, 0, 3);
    switch (payload_type) {
    case 0xABCDFFFF: {
        const char* frame_buffer = payload_buffer + 4;
        size_t frame_size = payload_size - 4;
        if (frame_size != 131096) {
            LOG4CXX_INFO(logger_, "got an image frame with wrong size " << frame_size << ", skip it.");
            return kSkipped;
        }
        uint64_t key = gDC.decode_byte<uint64_t>(frame_buffer, 12, 19);
        LOG4CXX_DEBUG(logger_, "received event with key " << key);
        // size_t index = hash_long_(key) % sender_count_;
        // LOG4CXX_DEBUG(logger_, "received an image frame with key: " << key);
        // if (sender_array_[index]->push(frame_buffer, frame_size)) {
        //     LOG4CXX_DEBUG(logger_, "pushed the image frame into sender[" << index << "].");
        //     return kProcessed;
        // } else {
        //     LOG4CXX_WARN(logger_, "sender[" << index << "] is stopped, close the connection.");
        //     return kFailed;
        // }
    } break;
    default:
        LOG4CXX_INFO(logger_, "got unknown payload, do nothing and jump it.");
        return kSkipped;
    }
}

uint32_t diffraflow::DspImgFrmConn::hash_long_(uint64_t value) { return (uint32_t)(value ^ (value >> 32)); }
