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

log4cxx::LoggerPtr diffraflow::GenericConnection::logger_ = log4cxx::Logger::getLogger("GenericConnection");

diffraflow::GenericConnection::GenericConnection(
    int sock_fd, uint32_t greet_hd, uint32_t recv_hd, uint32_t send_hd, size_t buff_sz) {
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

    network_metrics.total_sent_size = 0;
    network_metrics.total_sent_counts = 0;
    network_metrics.total_received_size = 0;
    network_metrics.total_received_counts = 0;
    network_metrics.total_processed_counts = 0;
    network_metrics.total_skipped_counts = 0;
}

diffraflow::GenericConnection::~GenericConnection() { delete[] buffer_; }

void diffraflow::GenericConnection::run() {
    if (start_connection_()) {
        while (!done_flag_ && do_receiving_and_processing_())
            ;
    }
    shutdown(client_sock_fd_, SHUT_RDWR);
    close(client_sock_fd_);
    done_flag_ = true;
    return;
}

bool diffraflow::GenericConnection::done() { return done_flag_; }

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
    uint32_t head = gDC.decode_byte<uint32_t>(buffer_, 0, 3);
    uint32_t size = gDC.decode_byte<uint32_t>(buffer_, 4, 7);
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
    after_connected_();
    return true;
}

void diffraflow::GenericConnection::after_connected_() { LOG4CXX_INFO(logger_, "connection ID: " << connection_id_); }

bool diffraflow::GenericConnection::do_receiving_and_processing_() {
    if (!NetworkUtils::receive_packet(
            client_sock_fd_, receiving_head_, buffer_, buffer_size_, buffer_limit_, logger_)) {
        return false;
    }

    network_metrics.total_received_size += buffer_limit_;
    network_metrics.total_received_counts += 1;

    // payload level process
    switch (process_payload_(buffer_, buffer_limit_)) {
    case kProcessed:
        network_metrics.total_processed_counts += 1;
        return true;
    case kSkipped:
        network_metrics.total_skipped_counts += 1;
        return true;
    case kFailed:
        return false;
    }
}

diffraflow::GenericConnection::ProcessRes diffraflow::GenericConnection::process_payload_(
    const char* payload_buffer, const size_t payload_size) {
    LOG4CXX_WARN(logger_, "function process_payload_() is used, but it is not implemented by subclass.");
    return kFailed;
}

bool diffraflow::GenericConnection::send_one_(const char* payload_head_buffer, const size_t payload_head_size,
    const char* payload_data_buffer, const size_t payload_data_size) {

    if (NetworkUtils::send_packet(client_sock_fd_, sending_head_, payload_head_buffer, payload_head_size,
            payload_data_buffer, payload_data_size, logger_)) {

        network_metrics.total_sent_size += (8 + payload_head_size + payload_data_size);
        network_metrics.total_sent_counts += 1;

        return true;
    } else {
        return false;
    }
}

bool diffraflow::GenericConnection::receive_one_(char* buffer, const size_t buffer_size, size_t& payload_size) {

    if (NetworkUtils::receive_packet(client_sock_fd_, receiving_head_, buffer, buffer_size, payload_size, logger_)) {

        network_metrics.total_received_size += 8 + payload_size;
        // 8 is the size of packet head
        network_metrics.total_received_counts += 1;

        return true;
    } else {
        return false;
    }
}

json::value diffraflow::GenericConnection::collect_metrics() {

    json::value network_metrics_json;
    network_metrics_json["total_sent_size"] = json::value::number(network_metrics.total_sent_size.load());
    network_metrics_json["total_sent_counts"] = json::value::number(network_metrics.total_sent_counts.load());
    network_metrics_json["total_received_size"] = json::value::number(network_metrics.total_received_size.load());
    network_metrics_json["total_received_counts"] = json::value::number(network_metrics.total_received_counts.load());
    network_metrics_json["total_processed_counts"] = json::value::number(network_metrics.total_processed_counts.load());
    network_metrics_json["total_skipped_counts"] = json::value::number(network_metrics.total_skipped_counts.load());

    json::value root_json;
    root_json["network_stats"] = network_metrics_json;
    root_json["connection_id"] = json::value::number(connection_id_);

    return root_json;
}