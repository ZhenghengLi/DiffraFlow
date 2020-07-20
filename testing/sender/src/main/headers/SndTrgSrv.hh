#ifndef __SndTrgSrv_H__
#define __SndTrgSrv_H__

#include "GenericServer.hh"

namespace diffraflow {

    class SndDatTran;

    class SndTrgSrv : public GenericServer {
    public:
        SndTrgSrv(string host, int port, SndDatTran* dat_tran);
        ~SndTrgSrv();

    protected:
        GenericConnection* new_connection_(int client_sock_fd);

    private:
        SndDatTran* data_transfer_;
    };

} // namespace diffraflow

#endif