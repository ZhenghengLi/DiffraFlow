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

void diffraflow::DspImgFrmConn::before_transferring_() {
    BOOST_LOG_TRIVIAL(info) << "connection ID: " << get_connection_id_();
}

bool diffraflow::DspImgFrmConn::do_transferring_() {
    const int slice_length = read(client_sock_fd_, buffer_ + slice_begin_, buffer_size_ - slice_begin_);
    if (slice_length == 0) {
        BOOST_LOG_TRIVIAL(info) << "socket " << client_sock_fd_ << " is closed.";
        return false;
    }
    const size_t limit = slice_begin_ + slice_length;
    size_t position = 0;
    while (true) {
        /////////////////////////////////////////////////////////////////
        // packet level process //
        /////////////////////////////////////////////////////////////////
        // continue to receive the data of head and size
        if (limit - position < 8) {
            shift_left_(position, limit);
            break;
        }
        // extract packet head and packet size
        uint32_t packet_head = gDC.decode_byte<uint32_t>(buffer_ + position, 0, 3);
        uint32_t packet_size = gDC.decode_byte<uint32_t>(buffer_ + position, 4, 7);
        position += 8;
        // head and size check for packet
        if (packet_head != 0xFFF22DDD) {
            BOOST_LOG_TRIVIAL(info) << "got wrong packet, close the connection.";
            return false;
        }
        if (packet_size > pkt_maxlen_) {
            BOOST_LOG_TRIVIAL(info) << "got too long packet, close the connection.";
            return false;
        }
        if (packet_size < 4) {
            BOOST_LOG_TRIVIAL(info) << "got too short packet, close the connection.";
            return false;
        }
        // continue to receive more data if reach half packet
        if (limit - position < packet_size) {
            position -= 8;
            shift_left_(position, limit);
            break;
        }
        // now have the whole packet from position, extract type and size of payload
        uint32_t payload_type = gDC.decode_byte<uint32_t>(buffer_ + position, 0, 3);
        const uint32_t payload_size = packet_size - 4;
        position += 4;

        //////////////////////////////////////////////////////////////////
        // payload level process //
        //////////////////////////////////////////////////////////////////
        // type and size check
        if (payload_type == 0xABCDFFFF && payload_size <= 8) {
            BOOST_LOG_TRIVIAL(info) << "got wrong image frame, close the connection.";
            return false;
        }

        // dispatch one image frame
        if (payload_type == 0xABCDFFFF) {
            int64_t identifier = gDC.decode_byte<int64_t>(buffer_ + position, 0, 7);
            int index = hash_long_(identifier) % sender_count_;
            BOOST_LOG_TRIVIAL(info) << "Send data with key: " << identifier;
            sender_array_[index]->push(buffer_ + position, payload_size);
        } else {
            BOOST_LOG_TRIVIAL(info) << "got unknown payload, do nothing and jump it.";
        }
        // jump to next packet
        position += payload_size;
        ///////////////////////////////////////////////////////////////////
    }
    return true;
}

int diffraflow::DspImgFrmConn::hash_long_(int64_t value) {
    return (int) (value ^ (value >> 32));
}
