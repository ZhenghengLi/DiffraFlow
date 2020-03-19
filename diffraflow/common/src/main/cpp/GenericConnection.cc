#include "GenericConnection.hh"
#include "Decoder.hh"
#include "PrimitiveSerializer.hh"

#include <cassert>
#include <unistd.h>
#include <netinet/in.h>
#include <algorithm>
#include <string.h>
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

using std::copy;

log4cxx::LoggerPtr diffraflow::GenericConnection::logger_
    = log4cxx::Logger::getLogger("GenericConnection");

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
    done_flag_ = false;
    connection_id_ = -1;
}

diffraflow::GenericConnection::~GenericConnection() {
    delete [] buffer_;
}

void diffraflow::GenericConnection::run() {
    if (start_connection_()) {
        before_receiving_();
        while (!done_flag_ && do_receiving_and_processing_());
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
    slice_begin_ = 0;
    while (true) {
        const int slice_length = read(client_sock_fd_, buffer_ + slice_begin_, buffer_size_ - slice_begin_);
        if (slice_length == 0) {
            LOG4CXX_INFO(logger_, "socket " << client_sock_fd_ << " is closed.");
            return false;
        }
        if (slice_begin_ + slice_length < 12) {
            slice_begin_ += slice_length;
        } else {
            break;
        }
    }
    uint32_t head  = gDC.decode_byte<uint32_t>(buffer_, 0, 3);
    uint32_t size  = gDC.decode_byte<uint32_t>(buffer_, 4, 7);
    connection_id_ = gDC.decode_byte<uint32_t>(buffer_, 8, 11);
    if (head != greeting_head_ || size != 4) {
        LOG4CXX_INFO(logger_, "got wrong greeting message, close the connection.");
        // send failure code which is 4321
        gPS.serializeValue<uint32_t>(4321, buffer_, 4);
        for (size_t pos = 0; pos < 4;) {
            int count = write(client_sock_fd_, buffer_ + pos, 4 - pos);
            if (count < 0) {
                LOG4CXX_WARN(logger_, "error found when sending failure code: " << strerror(errno));
                break;
            } else {
                pos += count;
            }
        }
        done_flag_ = false;
        return false;
    }
    // send success code which is 1234
    gPS.serializeValue<uint32_t>(1234, buffer_, 4);
    for (size_t pos = 0; pos < 4;) {
        int count = write(client_sock_fd_, buffer_ + pos, 4 - pos);
        if (count < 0) {
            LOG4CXX_WARN(logger_, "error found when sending success code: " << strerror(errno));
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

void diffraflow::GenericConnection::before_receiving_() {
    LOG4CXX_INFO(logger_, "connection ID: " << connection_id_);
}

bool diffraflow::GenericConnection::do_receiving_and_processing_() {
    const int slice_length = read(client_sock_fd_, buffer_ + slice_begin_, buffer_size_ - slice_begin_);
    if (slice_length == 0) {
        LOG4CXX_INFO(logger_, "socket " << client_sock_fd_ << " is closed.");
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
            LOG4CXX_INFO(logger_, "got wrong packet, close the connection.");
            return false;
        }
        if (packet_size > pkt_maxlen_) {
            LOG4CXX_INFO(logger_, "got too long packet, close the connection.");
            return false;
        }
        if (packet_size < 4) {
            LOG4CXX_INFO(logger_, "got too short packet, close the connection.");
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
        ProcessRes result = process_payload_(buffer_ + position, payload_size, payload_type);
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
    const char* payload_buffer, const uint32_t payload_size, const uint32_t payload_type) {
    LOG4CXX_WARN(logger_, "function process_payload_() is used, but it is not implemented by subclass.");
    return kStop;
}
