#include "DspImgFrmSrv.hh"
#include "DspSender.hh"
#include "DspImgFrmConn.hh"

diffraflow::DspImgFrmSrv::DspImgFrmSrv(string host, int port, DspSender** sender_arr, size_t sender_cnt)
    : GenericServer(host, port) {
    sender_array_ = sender_arr;
    sender_count_ = sender_cnt;
}

diffraflow::DspImgFrmSrv::~DspImgFrmSrv() {}

diffraflow::GenericConnection* diffraflow::DspImgFrmSrv::new_connection_(int client_sock_fd) {
    return new DspImgFrmConn(client_sock_fd, sender_array_, sender_count_);
}
