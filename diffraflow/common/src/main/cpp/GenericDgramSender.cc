#include "GenericDgramSender.hh"

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>

#define DGRAM_MSIZE 8210

log4cxx::LoggerPtr diffraflow::GenericDgramSender::logger_ = log4cxx::Logger::getLogger("GenericDgramSender");

diffraflow::GenericDgramSender::GenericDgramSender() {
    receiver_sock_host_ = "";
    receiver_sock_port_ = -1;
    sender_sock_fd_ = -1;
    memset(&receiver_addr_, 0, sizeof(receiver_addr_));
    // init metrics
    dgram_metrics.total_send_count = 0;
    dgram_metrics.total_succ_count = 0;
    dgram_metrics.total_error_count = 0;
    dgram_metrics.total_zero_count = 0;
    dgram_metrics.total_partial_count = 0;
}

diffraflow::GenericDgramSender::~GenericDgramSender() { close_sock(); }

string diffraflow::GenericDgramSender::get_receiver_address() {
    return receiver_sock_host_ + ":" + std::to_string(receiver_sock_port_);
}

bool diffraflow::GenericDgramSender::init_addr_sock(string host, int port) {
    // check
    if (sender_sock_fd_ >= 0) {
        LOG4CXX_WARN(
            logger_, "already initialized with receiver address " << receiver_sock_host_ << ":" << receiver_sock_port_);
        return true;
    }
    if (host.empty() || port < 0) {
        LOG4CXX_ERROR(logger_, "invalid address.");
        return false;
    }
    // store receiver address
    receiver_sock_host_ = host;
    receiver_sock_port_ = port;
    // prepare address
    addrinfo hints, *infoptr;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    int result = getaddrinfo(receiver_sock_host_.c_str(), NULL, &hints, &infoptr);
    if (result) {
        LOG4CXX_ERROR(logger_, "getaddrinfo: " << gai_strerror(result));
        return false;
    }
    ((sockaddr_in*)(infoptr->ai_addr))->sin_port = htons(receiver_sock_port_);
    memset(&receiver_addr_, 0, sizeof(receiver_addr_));
    memcpy(&receiver_addr_, infoptr->ai_addr, infoptr->ai_addrlen);
    freeaddrinfo(infoptr);

    // create socket
    sender_sock_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (sender_sock_fd_ < 0) {
        LOG4CXX_ERROR(logger_, "failed to create socket with error: " << strerror(errno));
        return false;
    }

    return true;
}

bool diffraflow::GenericDgramSender::send_datagram(const char* buffer, size_t len) {
    if (sender_sock_fd_ < 0) {
        LOG4CXX_ERROR(logger_, "sender is not initialized.");
        return false;
    }
    if (len > DGRAM_MSIZE) {
        LOG4CXX_ERROR(logger_, "size of datagram to send is larger than " << DGRAM_MSIZE);
        return false;
    }
    dgram_metrics.total_send_count++;
    int sndlen = sendto(sender_sock_fd_, buffer, len, 0, (struct sockaddr*)&receiver_addr_, sizeof(receiver_addr_));
    if (sndlen < 0) {
        dgram_metrics.total_error_count++;
        LOG4CXX_WARN(logger_, "found error when sending datagram: " << strerror(errno));
        return false;
    } else if (sndlen == 0) {
        dgram_metrics.total_zero_count++;
        LOG4CXX_WARN(logger_, "datagram is not sent.");
        return false;
    } else if (sndlen != len) {
        dgram_metrics.total_partial_count++;
        LOG4CXX_WARN(logger_, "partial datagram is sent.");
        return false;
    } else {
        dgram_metrics.total_succ_count++;
        LOG4CXX_DEBUG(logger_, "datagram is successfully sent.");
        return true;
    }
}

void diffraflow::GenericDgramSender::close_sock() {
    if (sender_sock_fd_ >= 0) {
        shutdown(sender_sock_fd_, SHUT_RDWR);
        close(sender_sock_fd_);
        sender_sock_fd_ = -1;
    }
}

json::value diffraflow::GenericDgramSender::collect_metrics() {

    json::value dgram_metrics_json;
    dgram_metrics_json["total_send_count"] = json::value::number(dgram_metrics.total_send_count.load());
    dgram_metrics_json["total_succ_count"] = json::value::number(dgram_metrics.total_succ_count.load());
    dgram_metrics_json["total_error_count"] = json::value::number(dgram_metrics.total_error_count.load());
    dgram_metrics_json["total_zero_count"] = json::value::number(dgram_metrics.total_zero_count.load());
    dgram_metrics_json["total_partial_count"] = json::value::number(dgram_metrics.total_partial_count.load());

    json::value root_json;
    root_json["dgram_stats"] = dgram_metrics_json;

    return root_json;
}
