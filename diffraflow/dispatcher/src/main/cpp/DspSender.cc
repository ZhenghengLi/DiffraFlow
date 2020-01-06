#include "DspSender.hh"
#include "PrimitiveSerializer.hh"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <string.h>
#include <algorithm>
#include <chrono>

#include <boost/log/trivial.hpp>

diffraflow::DspSender::DspSender(string hostname, int port, int id) {
    dest_host_ = hostname;
    dest_port_ = port;
    sender_id_ = id;
    buffer_size_ = 1024 * 1024 * 4 - 16; // 4 MiB - 64 B
    size_threshold_ = 1024 * 1024;  // 1 MiB
    time_threshold_ = 100; // 0.1 second
    buffer_A_ = new char[buffer_size_];
    buffer_A_limit_ = 0;
    buffer_A_imgct_ = 0;
    buffer_B_ = new char[buffer_size_];
    buffer_B_limit_ = 0;
    buffer_B_imgct_ = 0;
    buff_compr_A_ = new char[buffer_size_];
    buff_compr_B_ = new char[buffer_size_];
    client_sock_fd_ = -1;
    sending_thread_ = nullptr;
    run_flag_ = false;
}

diffraflow::DspSender::~DspSender() {
    delete [] buffer_A_;
    delete [] buffer_B_;
    delete [] buff_compr_A_;
    delete [] buff_compr_B_;
}

bool diffraflow::DspSender::connect_to_combiner() {
    addrinfo hints, *infoptr;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    int result = getaddrinfo(dest_host_.c_str(), NULL, &hints, &infoptr);
    if (result) {
        BOOST_LOG_TRIVIAL(error) << "getaddrinfo: " << gai_strerror(result);
        return false;
    }
    client_sock_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (client_sock_fd_ < 0) {
        BOOST_LOG_TRIVIAL(error) << "Socket creationg error";
        return false;
    }
    ((sockaddr_in*)(infoptr->ai_addr))->sin_port = htons(dest_port_);
    if (connect(client_sock_fd_, infoptr->ai_addr, infoptr->ai_addrlen)) {
        close_connection();
        BOOST_LOG_TRIVIAL(error) << "Connection to " << dest_host_ << ":" << dest_port_ << " failed.";
        return false;
    }
    freeaddrinfo(infoptr);
    // send greeting message for varification
    char buffer[12];
    gPS.serializeValue<uint32_t>(0xDDCC1234, buffer, 4);
    gPS.serializeValue<uint32_t>(4, buffer + 4, 4);
    gPS.serializeValue<int32_t>(sender_id_, buffer + 8, 4);
    for (size_t pos = 0; pos < 12;) {
        int count = write(client_sock_fd_, buffer + pos, 12 - pos);
        if (count > 0) {
            pos += count;
        } else {
            close_connection();
            BOOST_LOG_TRIVIAL(error) << "error found when doing the first write.";
            return false;
        }
    }
    for (size_t pos = 0; pos < 4;) {
        int count = read(client_sock_fd_, buffer + pos, 4 - pos);
        if (count > 0) {
            pos += count;
        } else {
            close_connection();
            BOOST_LOG_TRIVIAL(error) << "error found when doing the first read.";
            return false;
        }
    }
    int response_code = 0;
    gPS.deserializeValue<int32_t>(&response_code, buffer, 4);
    if (response_code != 1234) {
        close_connection();
        BOOST_LOG_TRIVIAL(error) << "Got wrong response code, close the connection.";
        return false;
    } else {
        BOOST_LOG_TRIVIAL(info) << "Successfully connected to combiner running on " << dest_host_ << ":" << dest_port_;
        return true;
    }
}

void diffraflow::DspSender::close_connection() {
    if (client_sock_fd_ >= 0) {
        close(client_sock_fd_);
        client_sock_fd_ = -1;
    }
}

void diffraflow::DspSender::push(const char* data, const size_t len) {
    if (!run_flag_) return;
    unique_lock<mutex> lk(mtx_);
    // wait when there is no enough space
    cv_push_.wait(lk, [&]() {return !run_flag_ || (len + 4 <= buffer_size_ - buffer_A_limit_);});
    if (!run_flag_) return;
    // image frame size
    gPS.serializeValue<uint32_t>(len, buffer_A_ + buffer_A_limit_, 4);
    // image frame data
    std::copy(data, data + len, buffer_A_ + buffer_A_limit_ + 4);
    buffer_A_limit_ += 4 + len;
    buffer_A_imgct_ += 1;
    // foreward limit and check size threshold
    if (buffer_A_limit_ > size_threshold_) {
        cv_swap_.notify_one();
    }
}

bool diffraflow::DspSender::swap_() {
    // swap buffer_A_ and buffer_B_ for async sending
    unique_lock<mutex> lk(mtx_);
    if (buffer_A_limit_ < size_threshold_) {
        cv_swap_.wait_for(lk, std::chrono::microseconds(time_threshold_));
    }
    if (buffer_A_limit_ > 0) {
        // do the swap
        char* tmp_buff = buffer_B_;
        buffer_B_ = buffer_A_;
        buffer_B_limit_ = buffer_A_limit_;
        buffer_B_imgct_ = buffer_A_imgct_;
        buffer_A_ = tmp_buff;
        buffer_A_limit_ = 0;
        buffer_A_imgct_ = 0;
        cv_push_.notify_one();
        return true;
    } else {
        return false;
    }
}

void diffraflow::DspSender::send_() {
    // send all data in buffer_B_
    if (buffer_B_limit_ > 0) {
        send_buffer_(buffer_B_, buff_compr_B_, buffer_B_limit_, buffer_B_imgct_);
        buffer_B_limit_ = 0;
        buffer_B_imgct_ = 0;
    }
}

void diffraflow::DspSender::send_remaining() {
    // do this only after stopping and before deleting
    lock_guard<mutex> lk(mtx_);
    // send all data in buffer_A
    if (buffer_A_limit_ > 0) {
        send_buffer_(buffer_A_, buff_compr_A_, buffer_A_limit_, buffer_A_imgct_);
        buffer_A_limit_ = 0;
        buffer_A_imgct_ = 0;
    }
}

void diffraflow::DspSender::send_buffer_(const char* buffer, char* buff_compr, const size_t limit, const size_t imgct) {
    // try to connect if lose connection
    if (client_sock_fd_ < 0) {
        if (connect_to_combiner()) {
            BOOST_LOG_TRIVIAL(info) << "reconnected to combiner.";
        } else {
            BOOST_LOG_TRIVIAL(warning) << "failed to reconnect to combiner, discard data in buffer.";
            return;
        }
    }
    // packet_head(4) | packet_size(4) | image_seq_head(4) | image_count(4) | image_seq_data
    // send head
    char head_buffer[16];
    gPS.serializeValue<uint32_t>(0xDDD22CCC, head_buffer, 4);
    gPS.serializeValue<uint32_t>(8 + limit, head_buffer + 4, 4);
    gPS.serializeValue<uint32_t>(0xABCDF8F8, head_buffer + 8, 4);
    gPS.serializeValue<uint32_t>(imgct, head_buffer + 12, 4);
    for (size_t pos = 0; pos < 16;) {
        int count = write(client_sock_fd_, head_buffer + pos, 16 - pos);
        if (count < 0) {
            BOOST_LOG_TRIVIAL(warning) << "error found when sending data: " << strerror(errno);
            close_connection();
            return;
        } else {
            pos += count;
        }
    }
    BOOST_LOG_TRIVIAL(info) << "done a write for head.";
    // compression can be done here: buffer -> buff_compr,
    // now directly send image sequence data without compression
    for (size_t pos = 0; pos < limit;) {
        int count = write(client_sock_fd_, buffer + pos, limit - pos);
        if (count < 0) {
            BOOST_LOG_TRIVIAL(warning) << "error found when sending data: " << strerror(errno);
            close_connection();
            return;
        } else {
            pos += count;
        }
    }
    BOOST_LOG_TRIVIAL(info) << "done a write for data.";
}

void diffraflow::DspSender::start() {
    if (run_flag_) return;
    run_flag_ = true;
    if (sending_thread_ != nullptr) return;
    sending_thread_ = new thread(
        [this]() {
            while (run_flag_) {
                if (swap_()) send_();
            }
        }
    );
}

void diffraflow::DspSender::stop() {
    if (!run_flag_) return;
    run_flag_ = false;
    if (sending_thread_ != nullptr) {
        sending_thread_->join();
        delete sending_thread_;
        sending_thread_ = nullptr;
    }
    cv_push_.notify_one();
}
