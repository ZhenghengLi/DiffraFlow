#include "GeneralServer.hh"
#include "GeneralConnection.hh"

#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <iostream>
#include <cassert>

using std::cout;
using std::cerr;
using std::endl;
using std::lock_guard;
using std::unique_lock;
using std::make_pair;

shine::GeneralServer::GeneralServer(int port) {
    server_sock_fd_ = -1;
    server_run_ = true;
    server_sock_port_ = port;
    server_sock_path_ = "";
    is_ipc_ = false;
    start_cleaner_();
}

shine::GeneralServer::GeneralServer(string sock_path) {
    server_sock_fd_ = -1;
    server_run_ = true;
    server_sock_port_ = 0;
    server_sock_path_ = sock_path;
    is_ipc_ = true;
    start_cleaner_();
}

shine::GeneralServer::~GeneralServer() {
    cleaner_run_ = false;
    cleaner_->join();
    stop();
    delete cleaner_;
}

void shine::GeneralServer::start_cleaner_() {
    dead_counts_ = 0;
    cleaner_run_ = true;
    cleaner_ = new thread(
        [this]() {
            while (cleaner_run_) clean_();
        }
    );
}

bool shine::GeneralServer::create_tcp_sock_() {
    sockaddr_in server_addr;
    server_sock_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock_fd_ < 0) {
        return false;
    }
    bzero(&server_addr, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(server_sock_port_);
    if (bind(server_sock_fd_, (sockaddr*) &server_addr, sizeof(server_addr)) < 0) {
        return false;
    }
    listen(server_sock_fd_, 5);
    return true;
}

bool shine::GeneralServer::create_ipc_sock_() {
    sockaddr_un server_addr;
    server_sock_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_sock_fd_ < 0) {
        return false;
    }
    bzero(&server_addr, sizeof(server_addr));
    server_addr.sun_family = AF_UNIX;
    strcpy(server_addr.sun_path, server_sock_path_.c_str());
    // remove sock_path if it exists
    unlink(server_sock_path_.c_str());
    // do the bind
    if (bind(server_sock_fd_, (sockaddr*) &server_addr, sizeof(server_addr)) < 0) {
        return false;
    }
    listen(server_sock_fd_, 5);
    return true;
}

int shine::GeneralServer::accept_client_() {
    int client_sock_fd = accept(server_sock_fd_, NULL, NULL);
    return client_sock_fd;
}

void shine::GeneralServer::serve() {
    if (is_ipc_) {
        if (create_ipc_sock_()) {
            cout << "Successfully created socket on unix socket file "
                 << server_sock_path_
                 << " with server_sock_fd "
                 << server_sock_fd_ << endl;
        } else {
            cerr << "Failed to create server socket on unix socket file "
                 << server_sock_path_
                 << "." << endl;
            return;
        }
    } else {
        if (create_tcp_sock_()) {
            cout << "Successfully created socket on port "
                 << server_sock_port_
                 << " with server_sock_fd "
                 << server_sock_fd_ << endl;
        } else {
            cerr << "Failed to create server socket on port "
                 << server_sock_port_
                 << "." << endl;
            return;
        }
    }
    while (server_run_) {
        cout << "Waitting for connection ..." << endl;
        int client_sock_fd = accept_client_();
        cout << "One connection is established with client_sock_fd " << client_sock_fd << endl;
        if (client_sock_fd < 0) return;
        if (!server_run_) {
            close(client_sock_fd);
            return;
        }
        GeneralConnection* conn_object = new_connection_(client_sock_fd);
        thread* conn_thread = new thread(
            [&, conn_object]() {
                conn_object->run();
                dead_counts_++;
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

void shine::GeneralServer::clean_() {
    unique_lock<mutex> lk(mtx_);
    cv_clean_.wait(lk, [&]() {return dead_counts_ > 0;});
    for (connListT_::iterator iter = connections_.begin(); iter != connections_.end();) {
        if (iter->first->done()) {
            iter->second->join();
            delete iter->second;
            delete iter->first;
            iter = connections_.erase(iter);
            dead_counts_--;
            cout << "delete one connection" << endl;
        } else {
            iter++;
        }
    }
}

void shine::GeneralServer::stop() {
    server_run_ = false;
    unique_lock<mutex> lk(mtx_);
    // close all connections
    for (connListT_::iterator iter = connections_.begin(); iter != connections_.end(); iter++) {
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
