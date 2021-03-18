#include "GenericDgramReceiver.hh"

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sched.h>

#include "ImageFrameDgram.hh"

using std::unique_lock;

log4cxx::LoggerPtr diffraflow::GenericDgramReceiver::logger_ = log4cxx::Logger::getLogger("GenericDgramReceiver");

diffraflow::GenericDgramReceiver::GenericDgramReceiver(string host, int port, int rcvbufsize) {
    receiver_sock_fd_ = -1;
    if (rcvbufsize < 512 * 1024) {
        receiver_sock_bs_ = 512 * 1024;
    } else if (rcvbufsize > 64 * 1024 * 1024) {
        receiver_sock_bs_ = 64 * 1024 * 1024;
    } else {
        receiver_sock_bs_ = rcvbufsize;
    }
    receiver_sock_host_ = host;
    receiver_sock_port_ = port;
    receiver_status_ = kNotStart;
    worker_thread_ = nullptr;
    worker_result_ = 0;
    memset(&receiver_addr_, 0, sizeof(receiver_addr_));
    memset(&sender_addr_, 0, sizeof(sender_addr_));
    sender_addr_len_ = 0;

    // init metrics
    dgram_metrics.total_recv_count = 0;
    dgram_metrics.total_recv_size = 0;
    dgram_metrics.total_error_count = 0;
    dgram_metrics.total_processed_count = 0;
    dgram_metrics.total_zero_count = 0;
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
    setsockopt(receiver_sock_fd_, SOL_SOCKET, SO_RCVBUF, (char*)&receiver_sock_bs_, sizeof(receiver_sock_bs_));

    // bind address
    if (bind(receiver_sock_fd_, (struct sockaddr*)&receiver_addr_, sizeof(receiver_addr_)) < 0) {
        LOG4CXX_ERROR(logger_, "bind: " << strerror(errno));
        return false;
    }

    return true;
}

void diffraflow::GenericDgramReceiver::run_() {
    if (receiver_status_ != kNotStart) {

        worker_result_ = 1;
        receiver_status_ = kStopped;
        cv_status_.notify_all();

        return;
    }
    if (create_udp_sock_()) {
        LOG4CXX_INFO(
            logger_, "Successfully created dgram socket on " << receiver_sock_host_ << ":" << receiver_sock_port_);
    } else {
        LOG4CXX_ERROR(
            logger_, "Failed to create dgram socket on " << receiver_sock_host_ << ":" << receiver_sock_port_);

        worker_result_ = 2;
        receiver_status_ = kStopped;
        cv_status_.notify_all();

        return;
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
        } else if (recvlen == 0) {
            LOG4CXX_WARN(logger_, "received a zero length datagram.");
            // do not stop, continue to receive next datagram
            dgram_metrics.total_zero_count++;
        } else {
            dgram_metrics.total_recv_size += recvlen;
            datagram->resize(recvlen);
            process_datagram_(datagram);
            dgram_metrics.total_processed_count++;
        }
    }

    worker_result_ = result;
    receiver_status_ = kStopped;
    cv_status_.notify_all();
}

void diffraflow::GenericDgramReceiver::process_datagram_(shared_ptr<vector<char>>& datagram) {
    // this method should be implemented by subclasses
}

bool diffraflow::GenericDgramReceiver::start(int cpu_id) {

    int num_cpus = std::thread::hardware_concurrency();
    if (cpu_id >= num_cpus) {
        LOG4CXX_ERROR(logger_, "CPU id (" << cpu_id << ") is too large, it should be smaller than " << num_cpus);
        return false;
    }

    if (!(receiver_status_ == kNotStart || receiver_status_ == kClosed)) {
        return false;
    }

    receiver_status_ = kNotStart;
    worker_thread_ = new thread(&GenericDgramReceiver::run_, this);

    bool cpu_bind_succ = true;
    bool status_running = false;

    if (cpu_id >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        int rc = pthread_setaffinity_np(worker_thread_->native_handle(), sizeof(cpu_set_t), &cpuset);
        if (rc == 0) {
            LOG4CXX_INFO(logger_, "successfully bind receiving thread on cpu " << cpu_id);
            cpu_bind_succ = true;
        } else {
            LOG4CXX_ERROR(logger_, "error calling pthread_setaffinity_np with error number: " << rc);
            cpu_bind_succ = false;
        }
    }

    unique_lock<mutex> ulk(mtx_status_);
    cv_status_.wait(ulk, [this]() { return receiver_status_ != kNotStart; });
    if (receiver_status_ == kRunning) {
        status_running = true;
    }

    if (cpu_bind_succ && status_running) {
        return true;
    } else {
        stop_and_close();
        return false;
    }
}

int diffraflow::GenericDgramReceiver::wait() {
    // check
    if (receiver_status_ == kNotStart) {
        return -1;
    }
    if (receiver_status_ == kClosed) {
        return -2;
    }
    unique_lock<mutex> ulk(mtx_status_);
    cv_status_.wait(ulk, [this]() { return receiver_status_ == kStopped; });
    return worker_result_;
}

int diffraflow::GenericDgramReceiver::stop_and_close() {
    // check
    if (receiver_status_ == kNotStart) {
        return -1;
    }
    if (receiver_status_ == kClosed) {
        return -2;
    }
    if (receiver_status_ == kRunning) {
        receiver_status_ = kStopping;
    }
    // shutdown and close socket
    if (receiver_sock_fd_ >= 0) {
        shutdown(receiver_sock_fd_, SHUT_RDWR);
        close(receiver_sock_fd_);
        receiver_sock_fd_ = -1;
    }
    // wait worker to finish
    int result = wait();
    // delete worker
    if (worker_thread_ != nullptr) {
        worker_thread_->join();
        delete worker_thread_;
        worker_thread_ = nullptr;
    }

    receiver_status_ = kClosed;

    LOG4CXX_INFO(logger_, "datagram receiver is closed.");
    return result;
}

json::value diffraflow::GenericDgramReceiver::collect_metrics() {

    json::value dgram_metrics_json;
    dgram_metrics_json["total_recv_count"] = json::value::number(dgram_metrics.total_recv_count.load());
    dgram_metrics_json["total_recv_size"] = json::value::number(dgram_metrics.total_recv_size.load());
    dgram_metrics_json["total_error_count"] = json::value::number(dgram_metrics.total_error_count.load());
    dgram_metrics_json["total_processed_count"] = json::value::number(dgram_metrics.total_processed_count.load());
    dgram_metrics_json["total_zero_count"] = json::value::number(dgram_metrics.total_zero_count.load());

    json::value root_json;
    root_json["dgram_stats"] = dgram_metrics_json;

    return root_json;
}
