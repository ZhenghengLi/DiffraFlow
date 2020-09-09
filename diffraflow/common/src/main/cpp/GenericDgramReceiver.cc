#include "GenericDgramReceiver.hh"

diffraflow::GenericDgramReceiver::GenericDgramReceiver(string host, int port) {
    receiver_sock_fd_ = -1;
    receiver_sock_host_ = host;
    receiver_sock_port_ = port;
}

diffraflow::GenericDgramReceiver::~GenericDgramReceiver() {}

bool diffraflow::GenericDgramReceiver::create_udp_sock_() {
    //
    return true;
}