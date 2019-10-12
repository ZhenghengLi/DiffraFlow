#include "ImageCacheServer.hpp"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <iostream>

using std::cout;
using std::cerr;
using std::endl;

shine::ImageCacheServer::ImageCacheServer(int p) {
    port_ = p;
    server_sock_fd_ = -1;
    run_ = true;
}

shine::ImageCacheServer::~ImageCacheServer() {

}

bool shine::ImageCacheServer::create_sock() {
    sockaddr_in server_addr, client_addr;
    server_sock_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock_fd_ < 0) {
        return false;
    }
    bzero( (char*) &server_addr, sizeof(server_addr) );
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port_);
    if (bind(server_sock_fd_, (sockaddr*) &server_addr, sizeof(server_addr)) < 0) {
        return false;
    }
    listen(server_sock_fd_, 5);
    return true;
}

int shine::ImageCacheServer::accept_client() {
    sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_sock_fd = accept(server_sock_fd_, (sockaddr*) &client_addr, &client_len);
    return client_sock_fd;
}

void shine::ImageCacheServer::serve() {
    while (run_) {
        if (server_sock_fd_ < 0) {
            cerr << "ERROR: socket is not open." << endl;
            return;
        }
        int client_sock_fd = accept_client();

    }
}

void shine::ImageCacheServer::clean() {

}
