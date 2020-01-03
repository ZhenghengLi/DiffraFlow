#include "DspImgFrmConn.hh"
#include "PrimitiveSerializer.hh"
#include "Decoder.hh"
#include "DspSender.hh"
#include <boost/log/trivial.hpp>

diffraflow::DspImgFrmConn::DspImgFrmConn(int sock_fd,
    DspSender** sender_arr, size_t sender_cnt):
    GenericConnection(sock_fd, 100 * 1024 * 1024, 1024 * 1024, 0xFFDD1234) {
    sender_array_ = sender_arr;
    sender_count_ = sender_cnt;

}

diffraflow::DspImgFrmConn::~DspImgFrmConn() {

}

diffraflow::DspImgFrmConn::ProcessRes diffraflow::DspImgFrmConn::process_payload_(
    const size_t payload_position, const uint32_t payload_size, const uint32_t payload_type) {
    if (payload_type == 0xABCDFFFF && payload_size <= 8) {
        BOOST_LOG_TRIVIAL(info) << "got wrong image frame, close the connection.";
        return kStop;
    }
    // dispatch one image frame
    if (payload_type == 0xABCDFFFF) {
        int64_t identifier = gDC.decode_byte<int64_t>(buffer_ + payload_position, 0, 7);
        int index = hash_long_(identifier) % sender_count_;
        BOOST_LOG_TRIVIAL(info) << "Send data with key: " << identifier;
        sender_array_[index]->push(buffer_ + payload_position, payload_size);
    } else {
        BOOST_LOG_TRIVIAL(info) << "got unknown payload, do nothing and jump it.";
    }
    return kContinue;
}

int diffraflow::DspImgFrmConn::hash_long_(int64_t value) {
    return (int) (value ^ (value >> 32));
}
