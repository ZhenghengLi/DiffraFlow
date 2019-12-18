#include "DispatchingServer.hh"
#include "Sender.hh"
#include "DispatchingConnection.hh"

diffraflow::DispatchingServer::DispatchingServer(int port,
    Sender** sender_arr, size_t sender_cnt): GeneralServer(port) {
    sender_array_ = sender_arr;
    sender_count_ = sender_cnt;
}

diffraflow::DispatchingServer::~DispatchingServer() {

}

diffraflow::GeneralConnection* diffraflow::DispatchingServer::new_connection_(int client_sock_fd) {
    return new DispatchingConnection(client_sock_fd, sender_array_, sender_count_);
}