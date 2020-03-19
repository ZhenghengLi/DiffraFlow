#include "GenericConnection.hh"
#include "Decoder.hh"
#include "PrimitiveSerializer.hh"
#include "NetworkUtils.hh"

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
    size_t buff_sz) {
    assert(sock_fd > 0);
    client_sock_fd_ = sock_fd;
    greeting_head_ = greet_hd;
    receiving_head_ = recv_hd;
    sending_head_ = send_hd;
    buffer_size_ = buff_sz;
    buffer_ = new char[buffer_size_];
    buffer_limit_ = 0;
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

bool diffraflow::GenericConnection::start_connection_() {
    // receive greeting
    for (size_t pos = 0; pos < 12;) {
        int count = read(client_sock_fd_, buffer_ + pos, 12 - pos);
        if (count < 0) {
            LOG4CXX_WARN(logger_, "error found when receiving data: " << strerror(errno));
            return false;
        } else if (count == 0) {
            LOG4CXX_INFO(logger_, "socket " << client_sock_fd_ << " is closed.");
            return false;
        } else {
            pos += count;
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
    return true;
}

void diffraflow::GenericConnection::before_receiving_() {
    LOG4CXX_INFO(logger_, "connection ID: " << connection_id_);
}

bool diffraflow::GenericConnection::do_receiving_and_processing_() {
    if (!NetworkUtils::receive_packet(client_sock_fd_, receiving_head_,
        buffer_, buffer_size_, buffer_limit_, logger_)) {
        return false;
    }
    // payload level process
    if (process_payload_(buffer_, buffer_limit_) ) {
        return true;
    } else {
        return false;
    }
}

bool diffraflow::GenericConnection::process_payload_(
    const char* payload_buffer, const size_t payload_size) {
    LOG4CXX_WARN(logger_, "function process_payload_() is used, but it is not implemented by subclass.");
    return false;
}

bool diffraflow::GenericConnection::send_one_(
    const char*    payload_head_buffer,
    const size_t   payload_head_size,
    const char*    payload_data_buffer,
    const size_t   payload_data_size) {

    if (NetworkUtils::send_packet(
        client_sock_fd_,
        sending_head_,
        payload_head_buffer,
        payload_head_size,
        payload_data_buffer,
        payload_data_size,
        logger_) ) {
        // note: here can accumulate metrics
        return true;
    } else {
        return false;
    }

}