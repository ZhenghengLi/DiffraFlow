#include "GenericDgramReceiver.hh"

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>

#include "ImageFrameDgram.hh"

using std::unique_lock;

log4cxx::LoggerPtr diffraflow::GenericDgramReceiver::logger_ = log4cxx::Logger::getLogger("GenericDgramReceiver");

diffraflow::GenericDgramReceiver::GenericDgramReceiver(string host, int port) {
    receiver_sock_fd_ = -1;
    receiver_sock_host_ = host;
    receiver_sock_port_ = port;
    receiver_status_ = kNotStart;
    memset(&receiver_addr_, 0, sizeof(receiver_addr_));
    memset(&sender_addr_, 0, sizeof(sender_addr_));
    sender_addr_len_ = 0;

    // init metrics
    dgram_metrics.total_recv_count = 0;
    dgram_metrics.total_recv_size = 0;
    dgram_metrics.total_error_count = 0;
    dgram_metrics.total_processed_count = 0;
}

diffraflow::GenericDgramReceiver::~GenericDgramReceiver() { stop_and_close(); }

bool diffraflow::GenericDgramReceiver::create_udp_sock_() {
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
    receiver_sock_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (receiver_sock_fd_ < 0) {
        LOG4CXX_ERROR(logger_, "failed to create socket with error: " << strerror(errno));
        return false;
    }

    // set larger receive buffer
    int rcvbufsize = 4194304; // 4 MiB
    setsockopt(receiver_sock_fd_, SOL_SOCKET, SO_RCVBUF, (char*)&rcvbufsize, sizeof(rcvbufsize));

    // bind address
    if (bind(receiver_sock_fd_, (struct sockaddr*)&receiver_addr_, sizeof(receiver_addr_)) < 0) {
        LOG4CXX_ERROR(logger_, "bind: " << strerror(errno));
        return false;
    }

    return true;
}

int diffraflow::GenericDgramReceiver::run_() {
    if (receiver_status_ != kNotStart) {
        return 1;
    }
    if (create_udp_sock_()) {
        LOG4CXX_INFO(
            logger_, "Successfully created dgram socket on " << receiver_sock_host_ << ":" << receiver_sock_port_);
    } else {
        LOG4CXX_ERROR(
            logger_, "Failed to create dgram socket on " << receiver_sock_host_ << ":" << receiver_sock_port_);
        return 2;
    }

    receiver_status_ = kRunning;
    cv_status_.notify_all();

    int result = 0;

    while (receiver_status_ == kRunning) {
        LOG4CXX_DEBUG(logger_, "waiting for datagram ...");
        shared_ptr<vector<char>> datagram = make_shared<vector<char>>(DGRAM_MSIZE);
        int recvlen = recvfrom(receiver_sock_fd_, datagram->data(), datagram->size(), 0,
            (struct sockaddr*)&sender_addr_, &sender_addr_len_);
        LOG4CXX_DEBUG(logger_, "received one datagram of size: " << recvlen);
        if (receiver_status_ != kRunning) break;
        dgram_metrics.total_recv_count++;
        if (recvlen < 0) {
            LOG4CXX_WARN(logger_, "found error when receiving datagram: " << strerror(errno));
            // do not stop, continue to receive next datagram
            dgram_metrics.total_error_count++;
        }
        if (recvlen > 0) {
            dgram_metrics.total_recv_size += recvlen;
            datagram->resize(recvlen);
            process_datagram_(datagram);
            dgram_metrics.total_processed_count++;
        }
    }

    return result;
}

void diffraflow::GenericDgramReceiver::process_datagram_(shared_ptr<vector<char>>& datagram) {
    // this method should be implemented by subclasses
}

bool diffraflow::GenericDgramReceiver::start() {
    if (!(receiver_status_ == kNotStart || receiver_status_ == kClosed)) {
        return false;
    }
    receiver_status_ = kNotStart;
    worker_ = async(&GenericDgramReceiver::run_, this);
    unique_lock<mutex> ulk(mtx_status_);
    cv_status_.wait(ulk, [this]() { return receiver_status_ != kNotStart; });
    if (receiver_status_ == kRunning) {
        return true;
    } else {
        return false;
    }
}

void diffraflow::GenericDgramReceiver::wait() {
    if (worker_.valid()) {
        worker_.wait();
    }
}

int diffraflow::GenericDgramReceiver::stop_and_close() {
    // check
    if (receiver_status_ == kNotStart) {
        return -1;
    }
    if (receiver_status_ == kClosed) {
        return -2;
    }
    receiver_status_ = kStopped;
    cv_status_.notify_all();
    // shutdown and close socket
    shutdown(receiver_sock_fd_, SHUT_RDWR);
    close(receiver_sock_fd_);
    receiver_sock_fd_ = -1;
    receiver_status_ = kClosed;
    cv_status_.notify_all();
    // wait worker to finish
    int result = -3;
    if (worker_.valid()) {
        result = worker_.get();
    }
    LOG4CXX_INFO(logger_, "datagram receiver is closed.");
    return result;
}

json::value diffraflow::GenericDgramReceiver::collect_metrics() {

    json::value dgram_metrics_json;
    dgram_metrics_json["total_recv_count"] = json::value::number(dgram_metrics.total_recv_count.load());
    dgram_metrics_json["total_recv_size"] = json::value::number(dgram_metrics.total_recv_size.load());
    dgram_metrics_json["total_error_count"] = json::value::number(dgram_metrics.total_error_count.load());
    dgram_metrics_json["total_processed_count"] = json::value::number(dgram_metrics.total_processed_count.load());

    json::value root_json;
    root_json["dgram_stats"] = dgram_metrics_json;

    return root_json;
}
