#include "GenericConnection.hh"
#include "Decoder.hh"

#include <cassert>
#include <unistd.h>
#include <netinet/in.h>
#include <algorithm>
#include <boost/log/trivial.hpp>

using std::copy;

diffraflow::GenericConnection::GenericConnection(int sock_fd,
    uint32_t greet_hd, uint32_t recv_hd, uint32_t send_hd,
    size_t buff_sz, size_t pkt_ml) {
    assert(sock_fd > 0);
    client_sock_fd_ = sock_fd;
    greeting_head_ = greet_hd;
    receiving_head_ = recv_hd;
    sending_head_ = send_hd;
    buffer_size_ = buff_sz;
    buffer_ = new char[buffer_size_];
    pkt_maxlen_ = pkt_ml;
    slice_begin_ = 0;
    buff_compr_ = new char[pkt_maxlen_];
    done_flag_ = false;
    connection_id_ = -1;
}

diffraflow::GenericConnection::~GenericConnection() {
    delete [] buffer_;
    delete [] buff_compr_;
}

void diffraflow::GenericConnection::run() {
    if (start_connection_()) {
        before_transferring_();
        while (!done_flag_ && do_transferring_());
    }
    close(client_sock_fd_);
    done_flag_ = true;
    return;
}

bool diffraflow::GenericConnection::done() {
    return done_flag_;
}

void diffraflow::GenericConnection::stop() {
    shutdown(client_sock_fd_, SHUT_RDWR);
    done_flag_ = true;
}

void diffraflow::GenericConnection::shift_left_(const size_t position, const size_t limit) {
    if (position > 0) {
        copy(buffer_ + position, buffer_ + limit, buffer_);
    }
    slice_begin_ = limit - position;
}

bool diffraflow::GenericConnection::start_connection_() {
    while (true) {
        const int slice_length = read(client_sock_fd_, buffer_ + slice_begin_, buffer_size_ - slice_begin_);
        if (slice_length == 0) {
            BOOST_LOG_TRIVIAL(info) << "socket " << client_sock_fd_ << " is closed.";
            return false;
        }
        if (slice_begin_ + slice_length < 12) {
            slice_begin_ += slice_length;
        } else {
            break;
        }
    }
    uint32_t success_code = htonl(1234);
    uint32_t failure_code = htonl(4321);
    uint32_t head = gDC.decode_byte<int32_t>(buffer_, 0, 3);
    uint32_t size = gDC.decode_byte<int32_t>(buffer_, 4, 7);
    if (head != greeting_head_ || size != 4) {
        BOOST_LOG_TRIVIAL(info) << "got wrong greeting message, close the connection.";
        for (size_t pos = 0; pos < 4;) {
            int count = write(client_sock_fd_, &failure_code, 4);
            if (count < 0) {
                BOOST_LOG_TRIVIAL(warning) << "error found when sending failure code: " << strerror(errno);
                break;
            } else {
                pos += count;
            }
        }
        done_flag_ = false;
        return false;
    }
    connection_id_ = gDC.decode_byte<int32_t>(buffer_, 8, 11);
    for (size_t pos = 0; pos < 4;) {
        int count = write(client_sock_fd_, &success_code, 4);
        if (count < 0) {
            BOOST_LOG_TRIVIAL(warning) << "error found when sending success code: " << strerror(errno);
            done_flag_ = false;
            return false;
        } else {
            pos += count;
        }
    }
    // ready for transferring data
    slice_begin_ = 0;
    return true;
}

void diffraflow::GenericConnection::before_transferring_() {
    BOOST_LOG_TRIVIAL(info) << "connection ID: " << connection_id_;
}

bool diffraflow::GenericConnection::do_transferring_() {
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
        if (packet_head != receiving_head_) {
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
        ProcessRes result = process_payload_(position, payload_size, payload_type);
        position += payload_size;
        switch (result) {
        case kContinue:
            // continue receiving data from current position
            continue;
        case kBreak:
            // slice_begin_ is reset, and should restart
            break;
        case kStop:
            // error found when processing payload, close the connection
            return false;
        }
        ///////////////////////////////////////////////////////////////////
    }
    return true;
}

diffraflow::GenericConnection::ProcessRes diffraflow::GenericConnection::process_payload_(
    const size_t payload_position, const uint32_t payload_size, const uint32_t payload_type) {
    BOOST_LOG_TRIVIAL(warning) << "function process_payload_() is used, but it is not implemented by subclass.";
    return kStop;
}
