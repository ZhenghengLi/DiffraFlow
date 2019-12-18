#include "DispatchingConnection.hh"
#include "PrimitiveSerializer.hh"
#include "Decoder.hh"
#include "Sender.hh"
#include <boost/log/trivial.hpp>

diffraflow::DispatchingConnection::DispatchingConnection(int sock_fd,
    Sender** sender_arr, size_t sender_cnt):
    GeneralConnection(sock_fd, 100 * 1024 * 1024, 1024 * 1024, 0xAABBCCDD) {
    sender_array_ = sender_arr;
    sender_count_ = sender_cnt;

}

diffraflow::DispatchingConnection::~DispatchingConnection() {

}

void diffraflow::DispatchingConnection::before_transferring_() {
    BOOST_LOG_TRIVIAL(info) << "connection ID: " << get_connection_id_();
}

bool diffraflow::DispatchingConnection::do_transferring_() {
    const int slice_length = read(client_sock_fd_, buffer_ + slice_begin_, buffer_size_ - slice_begin_);
    if (slice_length == 0) {
        BOOST_LOG_TRIVIAL(info) << "socket " << client_sock_fd_ << " is closed.";
        return false;
    }
    const size_t limit = slice_begin_ + slice_length;
    size_t position = 0;
    while (true) {
        if (limit - position < 8) {
            shift_left_(position, limit);
            break;
        }

        // read head and size
        uint32_t head = gDC.decode_byte<uint32_t>(buffer_ + position, 0, 3);
        position += 4;
        uint32_t size = gDC.decode_byte<uint32_t>(buffer_ + position, 0, 3);
        position += 4;

        // validation check for packet
        if (size > pkt_maxlen_) {
            BOOST_LOG_TRIVIAL(info) << "got too large packet, close the connection.";
            return false;
        }
        if (head == 0xABCDEEFF && size <= 8) {
            BOOST_LOG_TRIVIAL(info) << "got wrong image packet, close the connection.";
            return false;
        }

        // continue to receive more data
        if (limit - position < size) {
            position -= 8;
            shift_left_(position, limit);
            break;
        }

        // dispatch one packet
        uint64_t identifier;
        int index;

        switch (head) {
        case 0xABCDEEFF: // image data
            // extract identifier
            identifier = gDC.decode_byte<uint64_t>(buffer_ + position, 0, 7);
            index = hash_long_(identifier) % sender_count_;
            BOOST_LOG_TRIVIAL(info) << "Send data with key: " << identifier;
            sender_array_[index]->push(buffer_ + position, size);
            position += size;
            break;
        default:
            BOOST_LOG_TRIVIAL(info) << "got unknown packet, jump it.";
            position += size;
        }

    }
    return true;
}

int diffraflow::DispatchingConnection::hash_long_(int64_t value) {
    return (int) (value ^ (value >> 32));
}
