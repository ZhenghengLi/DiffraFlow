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
    buffer_size_ = 1024 * 1024 * 4; // 4 MiB
    size_threshold_ = 1024 * 1024;  // 1 MiB
    time_threshold_ = 100; // 0.1 second
    buffer_A_ = new char[buffer_size_];
    buffer_A_limit_ = 0;
    buffer_B_ = new char[buffer_size_];
    buffer_B_limit_ = 0;
    client_sock_fd_ = -1;
    sending_thread_ = nullptr;
}

diffraflow::DspSender::~DspSender() {
    delete [] buffer_A_;
    delete [] buffer_B_;
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
    if (connect(client_sock_fd_, infoptr->ai_addr, infoptr->ai_addrlen)) {
        close_connection();
        BOOST_LOG_TRIVIAL(error) << "Connection to " << dest_host_ << " failed.";
        return false;
    }
    freeaddrinfo(infoptr);
    // send greeting message for varification
    char buffer[12];
    gPS.serializeValue<uint32_t>(0xAAAABBBB, buffer, 4);
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
    if (response_code != 200) {
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
    unique_lock<mutex> lk(mtx_);
    // wait when there is no enough space
    cv_push_.wait(lk, [&]() {return len + 8 <= buffer_size_ - buffer_A_limit_;});
    // head
    gPS.serializeValue<uint32_t>(0xABCDEEFF, buffer_A_ + buffer_A_limit_, 4);
    // size
    gPS.serializeValue<uint32_t>(len, buffer_A_ + buffer_A_limit_ + 4, 4);
    // payload
    std::copy(data, data + len, buffer_A_ + buffer_A_limit_ + 8);
    buffer_A_limit_ += 8 + len;
    // foreward limit and check size threshold
    if (buffer_A_limit_ > size_threshold_) {
        cv_swap_.notify_one();
    }
}

bool diffraflow::DspSender::swap_() {
    unique_lock<mutex> lk(mtx_);
    if (buffer_A_limit_ < size_threshold_) {
        cv_swap_.wait_for(lk, std::chrono::microseconds(time_threshold_));
    }
    if (buffer_A_limit_ > 0) {
        // do the swap
        char*  tmp_buff = buffer_A_;
        size_t tmp_size = buffer_A_limit_;
        buffer_A_ = buffer_B_;
        buffer_A_limit_ = buffer_B_limit_;
        buffer_B_ = tmp_buff;
        buffer_B_limit_ = tmp_size;
        return true;
    } else {
        return false;
    }
}

void diffraflow::DspSender::send_() {
    // try to connect if lose connection
    if (client_sock_fd_ < 0) {
        if (connect_to_combiner()) {
            // BOOST_LOG_TRIVIAL(info) << "reconnected to combiner.";
        } else {
            // BOOST_LOG_TRIVIAL(warning) << "failed to reconnect to combiner, discard data in buffer.";
            buffer_B_limit_ = 0;
            return;
        }
    }
    for (size_t pos = 0; pos < buffer_B_limit_;) {
        int count = write(client_sock_fd_, buffer_B_ + pos, buffer_B_limit_ - pos);
        if (count == 0) { // need to test
            // BOOST_LOG_TRIVIAL(warning) << "connection is closed from the other side.";
            close_connection();
            buffer_B_limit_ = 0;
            return;
        } else if (count < 0) {
            // BOOST_LOG_TRIVIAL(warning) << "error found when sending data: " << strerror(errno);
            close_connection();
            buffer_B_limit_ = 0;
            return;
        } else {
            pos += count;
        }
    }
    BOOST_LOG_TRIVIAL(info) << "done a write.";
}

void diffraflow::DspSender::start() {
    run_flag_ = true;
    sending_thread_ = new thread(
        [this]() {
            while (run_flag_) {
                if (swap_()) send_();
            }
        }
    );
}

void diffraflow::DspSender::stop() {
    run_flag_ = false;
    if (sending_thread_ != nullptr) {
        sending_thread_->join();
        delete sending_thread_;
        sending_thread_ = nullptr;
    }
}
