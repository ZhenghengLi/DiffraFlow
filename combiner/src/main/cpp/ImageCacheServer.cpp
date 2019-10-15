#include "ImageCacheServer.hpp"
#include "ImageConnection.hpp"
#include "ImageCache.hpp"

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
using std::lock_guard;
using std::unique_lock;
using std::make_pair;

shine::ImageCacheServer::ImageCacheServer() {
    server_sock_fd_ = -1;
    server_run_ = true;
    clean_wait_time_ = 500;
    image_cache_ = new ImageCache();
    cleaner_run_ = true;
    cleaner_ = new thread(
        [this]() {
            while (cleaner_run_) clean_();
        }
    );
}

shine::ImageCacheServer::~ImageCacheServer() {
    cleaner_run_ = false;
    cleaner_->join();
    stop();
    delete cleaner_;
    delete image_cache_;
}

bool shine::ImageCacheServer::create_sock_(int port) {
    sockaddr_in server_addr, client_addr;
    server_sock_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock_fd_ < 0) {
        return false;
    }
    bzero( (char*) &server_addr, sizeof(server_addr) );
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);
    if (bind(server_sock_fd_, (sockaddr*) &server_addr, sizeof(server_addr)) < 0) {
        return false;
    }
    listen(server_sock_fd_, 5);
    return true;
}

int shine::ImageCacheServer::accept_client_() {
    sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_sock_fd = accept(server_sock_fd_, (sockaddr*) &client_addr, &client_len);
    return client_sock_fd;
}

void shine::ImageCacheServer::serve(int port) {
    if (!create_sock_(port)) {
        cout << "Failed to create server socket." << endl;
    }
    cout << "Successfully created socket on port " << port << " with server_sock_fd " << server_sock_fd_ << endl;
    while (server_run_) {
        cout << "Waitting for connection ..." << endl;
        int client_sock_fd = accept_client_();
        cout << "One connection is established with client_sock_fd " << client_sock_fd << endl;
        if (client_sock_fd < 0) return;
        if (!server_run_) {
            close(client_sock_fd);
            return;
        }
        ImageConnection* conn_object = new ImageConnection(client_sock_fd, image_cache_);
        thread* conn_thread = new thread(
            [&, conn_object]() {
                conn_object->run();
                cv_clean_.notify_one();
            }
        );
        {
            unique_lock<mutex> lk(mtx_);
            if (!server_run_) {
                conn_object->set_stop();
                conn_thread->join();
                delete conn_thread;
                delete conn_object;
                return;
            }
            connections_.push_back(make_pair(conn_object, conn_thread));
        }
    }
}

void shine::ImageCacheServer::clean_() {
    unique_lock<mutex> lk(mtx_);
    cv_clean_.wait_for(lk, std::chrono::milliseconds(clean_wait_time_));
    for (connList::iterator iter = connections_.begin(); iter != connections_.end();) {
        if (iter->first->done()) {
            iter->second->join();
            delete iter->second;
            delete iter->first;
            iter = connections_.erase(iter);
            cout << "delete one connection" << endl;
        } else {
            iter++;
        }
    }
}

void shine::ImageCacheServer::stop() {
    server_run_ = false;
    unique_lock<mutex> lk(mtx_);
    // close all connections
    for (connList::iterator iter = connections_.begin(); iter != connections_.end(); iter++) {
        iter->first->set_stop();
        iter->second->join();
        delete iter->second;
        delete iter->first;
        connections_.erase(iter);
    }
    // close the server socket
    if (server_sock_fd_ > 0) {
        close(server_sock_fd_);
    }
}
