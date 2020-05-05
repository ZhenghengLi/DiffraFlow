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
        if (payload_size <= 8) {
            LOG4CXX_INFO(logger_, "got wrong image frame, close the connection.");
            return kFailed;
        }
        uint64_t identifier = gDC.decode_byte<uint64_t>(payload_buffer, 4, 11);
        size_t index = hash_long_(identifier) % sender_count_;
        LOG4CXX_DEBUG(logger_, "Send data with key: " << identifier);
        if (sender_array_[index]->push(payload_buffer + 4, payload_size - 4)) {
            LOG4CXX_DEBUG(logger_, "pushed one image frame into sender.");
            return kProcessed;
        } else {
            LOG4CXX_WARN(logger_, "sender[" << index << "] is stopped, close the connection.");
            return kFailed;
        }
    }
    default:
        LOG4CXX_INFO(logger_, "got unknown payload, do nothing and jump it.");
        return kSkipped;
    }
}

uint32_t diffraflow::DspImgFrmConn::hash_long_(uint64_t value) { return (uint32_t)(value ^ (value >> 32)); }
