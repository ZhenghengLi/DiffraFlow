#include "GenericServer.hh"
#include "GenericConnection.hh"

#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <iostream>
#include <cassert>
#include <boost/log/trivial.hpp>

using std::lock_guard;
using std::unique_lock;
using std::make_pair;

diffraflow::GenericServer::GenericServer(int port) {
    server_sock_fd_ = -1;
    server_run_ = false;
    server_sock_port_ = port;
    server_sock_path_ = "";
    is_ipc_ = false;
}

diffraflow::GenericServer::GenericServer(string sock_path) {
    server_sock_fd_ = -1;
    server_run_ = false;
    server_sock_port_ = 0;
    server_sock_path_ = sock_path;
    is_ipc_ = true;
}

diffraflow::GenericServer::~GenericServer() {
    stop();
}

void diffraflow::GenericServer::start_cleaner_() {
    dead_counts_ = 0;
    cleaner_run_ = true;
    cleaner_ = new thread(
        [this]() {
            while (cleaner_run_) clean_();
        }
    );
}

void diffraflow::GenericServer::stop_cleaner_() {
    cleaner_run_ = false;
    cv_clean_.notify_one();
    cleaner_->join();
    delete cleaner_;
}

bool diffraflow::GenericServer::create_tcp_sock_() {
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

bool diffraflow::GenericServer::create_ipc_sock_() {
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

int diffraflow::GenericServer::accept_client_() {
    int client_sock_fd = accept(server_sock_fd_, NULL, NULL);
    return client_sock_fd;
}

void diffraflow::GenericServer::serve() {
    if (server_run_) return;
    server_run_ = true;
    if (is_ipc_) {
        if (create_ipc_sock_()) {
            BOOST_LOG_TRIVIAL(info)
                << "Successfully created socket on unix socket file "
                << server_sock_path_
                << " with server_sock_fd "
                << server_sock_fd_;
        } else {
            BOOST_LOG_TRIVIAL(error)
                << "Failed to create server socket on unix socket file "
                << server_sock_path_
                << ".";
            server_run_ = false;
            return;
        }
    } else {
        if (create_tcp_sock_()) {
            BOOST_LOG_TRIVIAL(info)
                << "Successfully created socket on port "
                << server_sock_port_
                << " with server_sock_fd "
                << server_sock_fd_;
        } else {
            BOOST_LOG_TRIVIAL(error)
                << "Failed to create server socket on port "
                << server_sock_port_
                << ".";
                server_run_ = false;
            return;
        }
    }
    // start cleaner only when server socket is successfully opened and is accepting connections.
    start_cleaner_();
    // start accepting clients
    while (server_run_) {
        BOOST_LOG_TRIVIAL(info) << "Waitting for connection ...";
        int client_sock_fd = accept_client_();
        if (client_sock_fd < 0) {
            if (server_run_) BOOST_LOG_TRIVIAL(error) << "got wrong client_sock_fd when server is running.";
            return;
        }
        BOOST_LOG_TRIVIAL(info) << "One connection is established with client_sock_fd " << client_sock_fd;
        if (!server_run_) {
            close(client_sock_fd);
            return;
        }
        GenericConnection* conn_object = new_connection_(client_sock_fd);
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
                conn_object->stop();
                conn_thread->join();
                delete conn_thread;
                delete conn_object;
                return;
            }
            connections_.push_back(make_pair(conn_object, conn_thread));
        }
    }
}

void diffraflow::GenericServer::clean_() {
    if (!cleaner_run_) return;
    unique_lock<mutex> lk(mtx_);
    cv_clean_.wait(lk, [&]() {return (!cleaner_run_ || dead_counts_ > 0);});
    for (connListT_::iterator iter = connections_.begin(); iter != connections_.end();) {
        if (iter->first->done()) {
            iter->second->join();
            delete iter->second;
            delete iter->first;
            iter = connections_.erase(iter);
            dead_counts_--;
            BOOST_LOG_TRIVIAL(info) << "delete one connection";
        } else {
            iter++;
        }
    }
}

void diffraflow::GenericServer::stop() {
    if (!server_run_) return;
    server_run_ = false;
    if (server_sock_fd_ < 0) return;
    // shutdown server socket in case new connection come in
    // SHUT_RD means further receptions will be disallowed.
    // so here only stop accepting, the opened connections are still alive.
    shutdown(server_sock_fd_, SHUT_RD);
    if (is_ipc_) {
        unlink(server_sock_path_.c_str());
    }
    // stop cleaner, then lock is not needed.
    stop_cleaner_();
    // close and delete all running connections
    // cleaner is stopped and no new connection can come in, so doing this without lock is safe.
    for (connListT_::iterator iter = connections_.begin(); iter != connections_.end();) {
        iter->first->stop();
        iter->second->join();
        delete iter->second;
        delete iter->first;
        iter = connections_.erase(iter);
    }
    // release socket resource
    close(server_sock_fd_);
    BOOST_LOG_TRIVIAL(info) << "server is closed.";
}
