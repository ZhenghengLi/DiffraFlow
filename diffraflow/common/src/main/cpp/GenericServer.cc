#include "GenericServer.hh"
#include "GenericConnection.hh"

#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <iostream>
#include <cassert>
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

using std::lock_guard;
using std::unique_lock;
using std::make_pair;

log4cxx::LoggerPtr diffraflow::GenericServer::logger_
    = log4cxx::Logger::getLogger("GenericServer");

diffraflow::GenericServer::GenericServer(string host, int port, size_t max_conn) {
    server_sock_fd_ = -1;
    server_run_ = false;
    server_sock_host_ = host;
    server_sock_port_ = port;
    server_sock_path_ = "";
    is_ipc_ = false;
    max_conn_counts_ = max_conn;
}

diffraflow::GenericServer::GenericServer(string sock_path, size_t max_conn) {
    server_sock_fd_ = -1;
    server_run_ = false;
    server_sock_host_ = "";
    server_sock_port_ = 0;
    server_sock_path_ = sock_path;
    is_ipc_ = true;
    max_conn_counts_ = max_conn;
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
    // prepare address
    addrinfo hints, *infoptr;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    int result = getaddrinfo(server_sock_host_.c_str(), NULL, &hints, &infoptr);
    if (result) {
        LOG4CXX_ERROR(logger_, "getaddrinfo: " << gai_strerror(result));
        return false;
    }
    ((sockaddr_in*)(infoptr->ai_addr))->sin_port = htons(server_sock_port_);
    // create socket
    server_sock_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock_fd_ < 0) {
        freeaddrinfo(infoptr);
        return false;
    }
    if (bind(server_sock_fd_, infoptr->ai_addr, infoptr->ai_addrlen) < 0) {
        LOG4CXX_ERROR(logger_, "bind: " << strerror(errno));
        freeaddrinfo(infoptr);
        return false;
    }
    listen(server_sock_fd_, 5);
    freeaddrinfo(infoptr);
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

int diffraflow::GenericServer::serve_() {
    if (server_run_) return 10;
    if (is_ipc_) {
        if (create_ipc_sock_()) {
            LOG4CXX_INFO(logger_,
                "Successfully created socket on unix socket file "
                << server_sock_path_
                << " with server_sock_fd " << server_sock_fd_ << ".");
        } else {
            LOG4CXX_ERROR(logger_,
                "Failed to create server socket on unix socket file "
                << server_sock_path_ << ".");
            return 11;
        }
    } else {
        if (create_tcp_sock_()) {
            LOG4CXX_INFO(logger_,
                "Successfully created socket on "
                << server_sock_host_ << ":" << server_sock_port_
                << " with server_sock_fd " << server_sock_fd_ << ".");
        } else {
            LOG4CXX_ERROR(logger_,
                "Failed to create server socket on "
                << server_sock_host_ << ":" << server_sock_port_ << ".");
            return 12;
        }
    }
    server_run_ = true;
    dead_counts_ = 0;
    int result = 0;
    // start cleaner only when server socket is successfully opened and is accepting connections.
    start_cleaner_();
    // start accepting clients
    while (server_run_) {
        LOG4CXX_INFO(logger_, "Waitting for connection ...");
        int client_sock_fd = accept_client_();
        if (client_sock_fd < 0) {
            if (server_run_) {
                LOG4CXX_ERROR(logger_, "got wrong client_sock_fd when server is running.");
                result = 21;
            }
            break;
        }
        LOG4CXX_INFO(logger_, "One connection is established with client_sock_fd " << client_sock_fd);
        if (!server_run_) {
            shutdown(client_sock_fd, SHUT_RDWR);
            close(client_sock_fd);
            break;
        }
        {
            lock_guard<mutex> lk(mtx_conn_);
            if (connections_.size() >= max_conn_counts_) {
                LOG4CXX_INFO(logger_, "The allowed number of connections reached maximum which is " << max_conn_counts_);
                shutdown(client_sock_fd, SHUT_RDWR);
                close(client_sock_fd);
                continue;
            }
            GenericConnection* conn_object = new_connection_(client_sock_fd);
            thread* conn_thread = new thread(
                [&, conn_object]() {
                    conn_object->run();
                    dead_counts_++;
                    cv_clean_.notify_one();
                }
            );
            if (server_run_) {
                connections_.push_back(make_pair(conn_object, conn_thread));
            } else {
                conn_object->stop();
                conn_thread->join();
                delete conn_thread;
                delete conn_object;
            }
        }
    }
    // return
    return result;
}

void diffraflow::GenericServer::clean_() {
    if (!cleaner_run_) return;
    unique_lock<mutex> lk(mtx_conn_);
    cv_clean_.wait(lk, [&]() {return (!cleaner_run_ || dead_counts_ > 0);});
    if (!cleaner_run_) return;
    for (connListT_::iterator iter = connections_.begin(); iter != connections_.end();) {
        if (iter->first->done()) {
            iter->second->join();
            delete iter->second;
            delete iter->first;
            iter = connections_.erase(iter);
            dead_counts_--;
            LOG4CXX_INFO(logger_, "delete one connection");
        } else {
            ++iter;
        }
    }
}

void diffraflow::GenericServer::start() {
    worker_ = async(&GenericServer::serve_, this);
}

void diffraflow::GenericServer::wait() {
    if (worker_.valid()) {
        worker_.wait();
    }
}

int diffraflow::GenericServer::get() {
    if (worker_.valid()) {
        return worker_.get();
    } else {
        return -1;
    }
}

void diffraflow::GenericServer::stop() {
    if (!server_run_) return;
    server_run_ = false;
    // shutdown server socket in case new connection come in.
    // SHUT_RD means further receptions will be disallowed,
    // so here only stop accepting, the opened connections are still alive.
    shutdown(server_sock_fd_, SHUT_RD);
    if (is_ipc_) {
        unlink(server_sock_path_.c_str());
    }
    // stop cleaner
    stop_cleaner_();
    // close and delete all running connections
    {
        lock_guard<mutex> lg(mtx_conn_);
        for (connListT_::iterator iter = connections_.begin(); iter != connections_.end();) {
            iter->first->stop();
            iter->second->join();
            delete iter->second;
            delete iter->first;
            iter = connections_.erase(iter);
            dead_counts_--;
        }
    }
    // release socket resource
    close(server_sock_fd_);
    // wait server_() to finish
    wait();

    LOG4CXX_INFO(logger_, "server is closed.");
}

json::value diffraflow::GenericServer::collect_metrics() {
    lock_guard<mutex> lk(mtx_conn_);
    json::value connection_metrics_json;
    int array_index = 0;
    for (connListT_::iterator iter = connections_.begin(); iter != connections_.end(); ++iter) {
        if (!iter->first->done()) {
            connection_metrics_json[array_index++] = iter->first->collect_metrics();
        }
    }
    return connection_metrics_json;
}