#include "SndTrgSrv.hh"
#include "SndDatTran.hh"
#include "SndTrgConn.hh"

diffraflow::SndTrgSrv::SndTrgSrv(string host, int port, SndDatTran* dat_tran) : GenericServer(host, port, 1) {
    data_transfer_ = dat_tran;
}

diffraflow::SndTrgSrv::~SndTrgSrv() {}

diffraflow::GenericConnection* diffraflow::SndTrgSrv::new_connection_(int client_sock_fd) {
    return new SndTrgConn(client_sock_fd, data_transfer_);
}