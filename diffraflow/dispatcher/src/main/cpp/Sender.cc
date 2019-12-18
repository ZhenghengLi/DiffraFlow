#include "Sender.hh"
#include "PrimitiveSerializer.hh"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <string.h>
#include <algorithm>
#include <chrono>

#include <boost/log/trivial.hpp>

diffraflow::Sender::Sender(string hostname, int port, int id) {
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
}

diffraflow::Sender::~Sender() {
    delete [] buffer_A_;
    delete [] buffer_B_;
}

bool diffraflow::Sender::connect_to_combiner() {
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
        close(client_sock_fd_);
        client_sock_fd_ = -1;
        BOOST_LOG_TRIVIAL(error) << "Connection to " << dest_host_ << " failed.";
        return false;
    }
    freeaddrinfo(infoptr);
    // send greeting message for varification
    char buffer[12];
    gPS.serializeValue<uint32_t>(0xAAAABBBB, buffer, 12);
    gPS.serializeValue<uint32_t>(4, buffer + 4, 8);
    gPS.serializeValue<int32_t>(sender_id_, buffer + 8, 4);
    write(client_sock_fd_, buffer, 12);
    read(client_sock_fd_, buffer, 12);
    int response_code = 0;
    gPS.deserializeValue<int32_t>(&response_code, buffer, 12);
    if (response_code != 200) {
        close(client_sock_fd_);
        client_sock_fd_ = -1;
        BOOST_LOG_TRIVIAL(error) << "Got wrong response code, close the connection.";
        return false;
    } else {
        BOOST_LOG_TRIVIAL(info) << "Successfully connectec to Combiner server " << dest_host_ << ":" << dest_port_;
        return true;
    }
}

void diffraflow::Sender::close_connection() {
    if (client_sock_fd_ >= 0) {
        close(client_sock_fd_);
        client_sock_fd_ = -1;
    }
}

void diffraflow::Sender::push(long key, char* data, size_t len) {
    unique_lock<mutex> lk(mtx_);
    cv_push_.wait(lk, [&]() {return len <= buffer_size_ - buffer_A_limit_;});
    std::copy(data, data + len, buffer_A_);
    buffer_A_limit_ += len;
    if (buffer_A_limit_ > size_threshold_) {
        cv_swap_.notify_one();
    }
}

bool diffraflow::Sender::swap_() {
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

void diffraflow::Sender::send_() {
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
        if (count == 0) {
            // BOOST_LOG_TRIVIAL(warning) << "connection is closed from the other side.";
            close_connection();
            buffer_B_limit_ = 0;
            return;
        } else if (count < 0) {
            BOOST_LOG_TRIVIAL(warning) << "error found when sending data: " << strerror(errno);
            buffer_B_limit_ = 0;
            return;
        } else {
            pos += count;
        }
    }
}

void diffraflow::Sender::run_() {
    while (run_flag_) {
        if (swap_()) send_();
    }
}

void diffraflow::Sender::start() {

}

void diffraflow::Sender::stop() {

}
