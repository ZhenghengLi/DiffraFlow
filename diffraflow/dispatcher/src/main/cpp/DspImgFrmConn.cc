#include "DspImgFrmConn.hh"
#include "PrimitiveSerializer.hh"
#include "Decoder.hh"
#include "DspSender.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

log4cxx::LoggerPtr diffraflow::DspImgFrmConn::logger_
    = log4cxx::Logger::getLogger("DspImgFrmConn");

diffraflow::DspImgFrmConn::DspImgFrmConn(int sock_fd,
    DspSender** sender_arr, size_t sender_cnt):
    GenericConnection(sock_fd, 0xFFDD1234, 0xFFF22DDD, 0xDDD22FFF, 1024 * 1024) {
    sender_array_ = sender_arr;
    sender_count_ = sender_cnt;
}

diffraflow::DspImgFrmConn::~DspImgFrmConn() {

}

bool diffraflow::DspImgFrmConn::process_payload_(
    const char* payload_buffer, const size_t payload_size) {
    // extract payload type
    uint32_t payload_type = gDC.decode_byte<uint32_t>(payload_buffer, 0, 3);
    switch (payload_type) {
    case 0xABCDFFFF:
        {
            if (payload_size <= 8) {
                LOG4CXX_INFO(logger_, "got wrong image frame, close the connection.");
                return false;
            }
            int64_t identifier = gDC.decode_byte<int64_t>(payload_buffer, 4, 11);
            int index = hash_long_(identifier) % sender_count_;
            LOG4CXX_INFO(logger_, "Send data with key: " << identifier);
            sender_array_[index]->push(payload_buffer + 4, payload_size - 4);
            return true;
        }
    default:
        LOG4CXX_INFO(logger_, "got unknown payload, do nothing and jump it.");
        return true;
    }
}

int diffraflow::DspImgFrmConn::hash_long_(int64_t value) {
    return (int) (value ^ (value >> 32));
}
