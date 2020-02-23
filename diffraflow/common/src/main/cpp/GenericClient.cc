#include "GenericClient.hh"
#include "PrimitiveSerializer.hh"

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <string.h>

log4cxx::LoggerPtr diffraflow::GenericClient::logger_
    = log4cxx::Logger::getLogger("GenericClient");

diffraflow::GenericClient::GenericClient(string hostname, int port, int id,
    uint32_t greet_hd, uint32_t send_hd) {
    dest_host_ = hostname;
    dest_port_ = port;
    client_id_ = id;
    greeting_head_ = greet_hd;
    sending_head_ = send_hd;
    client_sock_fd_ = -1;
}

diffraflow::GenericClient::~GenericClient() {
    close_connection();
}

bool diffraflow::GenericClient::connect_to_server() {
    addrinfo hints, *infoptr;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    int result = getaddrinfo(dest_host_.c_str(), NULL, &hints, &infoptr);
    if (result) {
        LOG4CXX_ERROR(logger_, "getaddrinfo: " << gai_strerror(result));
        return false;
    }
    client_sock_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (client_sock_fd_ < 0) {
        LOG4CXX_ERROR(logger_, "Socket creationg error");
        return false;
    }
    ((sockaddr_in*)(infoptr->ai_addr))->sin_port = htons(dest_port_);
    if (connect(client_sock_fd_, infoptr->ai_addr, infoptr->ai_addrlen)) {
        close_connection();
        LOG4CXX_ERROR(logger_, "Connection to " << dest_host_ << ":" << dest_port_ << " failed.");
        return false;
    }
    freeaddrinfo(infoptr);
    // send greeting message for varification
    char buffer[12];
    gPS.serializeValue<uint32_t>(greeting_head_, buffer, 4);
    gPS.serializeValue<uint32_t>(4, buffer + 4, 4);
    gPS.serializeValue<int32_t>(client_id_, buffer + 8, 4);
    for (size_t pos = 0; pos < 12;) {
        int count = write(client_sock_fd_, buffer + pos, 12 - pos);
        if (count > 0) {
            pos += count;
        } else {
            close_connection();
            LOG4CXX_ERROR(logger_, "error found when doing the first write.");
            return false;
        }
    }
    for (size_t pos = 0; pos < 4;) {
        int count = read(client_sock_fd_, buffer + pos, 4 - pos);
        if (count > 0) {
            pos += count;
        } else {
            close_connection();
            LOG4CXX_ERROR(logger_, "error found when doing the first read.");
            return false;
        }
    }
    int response_code = 0;
    gPS.deserializeValue<int32_t>(&response_code, buffer, 4);
    if (response_code != 1234) {
        close_connection();
        LOG4CXX_ERROR(logger_, "Got wrong response code, close the connection.");
        return false;
    } else {
        LOG4CXX_INFO(logger_, "Successfully connected to server running on " << dest_host_ << ":" << dest_port_);
        return true;
    }
}

void diffraflow::GenericClient::close_connection() {
    if (client_sock_fd_ >= 0) {
        close(client_sock_fd_);
        client_sock_fd_ = -1;
    }
}