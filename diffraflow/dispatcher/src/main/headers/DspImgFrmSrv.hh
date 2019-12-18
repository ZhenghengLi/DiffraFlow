#ifndef DspImgFrmSrv_H
#define DspImgFrmSrv_H

#include "GeneralServer.hh"

#include <cstddef>
#include <vector>
#include <string>

using std::vector;
using std::string;

namespace diffraflow {

    class DspSender;

    class DspImgFrmSrv: public GeneralServer {
    public:
        DspImgFrmSrv(int port, DspSender** sender_arr, size_t sender_cnt);
        ~DspImgFrmSrv();

    protected:
        GeneralConnection* new_connection_(int client_sock_fd);

    private:
        DspSender** sender_array_;
        size_t      sender_count_;
    };
}

#endif
