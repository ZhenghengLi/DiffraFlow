#include "Sender.hh"
#include "PrimitiveSerializer.hh"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <string.h>

#include <boost/log/trivial.hpp>

diffraflow::Sender::Sender(string hostname, int port, int id, size_t buff_size) {
    dest_host_ = hostname;
    dest_port_ = port;
    sender_id_ = id;
    buffer_size_ = buff_size;
    buffer_A_ = new char[buffer_size_];
    buffer_A_pos_ = 0;
    buffer_B_ = new char[buffer_size_];
    buffer_B_pos_ = 0;
    client_sock_fd_ = 0;
}

diffraflow::Sender::~Sender() {
    delete [] buffer_A_;
    delete [] buffer_B_;
}

bool diffraflow::Sender::conn() {
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
        client_sock_fd_ = 0;
        BOOST_LOG_TRIVIAL(error) << "Got wrong response code, close the connection.";
        return false;
    } else {
        BOOST_LOG_TRIVIAL(info) << "Successfully connectec to Combiner server " << dest_host_ << ":" << dest_port_;
        return true;
    }
}

void diffraflow::Sender::push(long key, char* data, size_t len) {

}

void diffraflow::Sender::swap() {

}

void diffraflow::Sender::start() {

}

void diffraflow::Sender::stop() {

}

void diffraflow::Sender::run_() {

}

void diffraflow::Sender::send_() {

}
