#include "DispatchingConnection.hh"

diffraflow::DispatchingConnection::DispatchingConnection(int sock_fd,
    Sender** sender_arr, size_t sender_cnt):
    GeneralConnection(sock_fd, 100 * 1024 * 1024, 1024 * 1024, 0xAABBCCDD) {
    sender_array_ = sender_arr;
    sender_count_ = sender_cnt;

}

diffraflow::DispatchingConnection::~DispatchingConnection() {

}

void diffraflow::DispatchingConnection::before_transferring_() {

}

bool diffraflow::DispatchingConnection::do_transferring_() {

    return true;
}
