#ifndef CmbImgFrmSrv_H
#define CmbImgFrmSrv_H

#include "GeneralServer.hh"

namespace diffraflow {

    class CmbImgFrmConn;
    class CmbImgCache;

    class CmbImgFrmSrv: public GeneralServer {
    private:
        CmbImgCache* image_cache_;

    protected:
        GeneralConnection* new_connection_(int client_sock_fd);

    public:
        CmbImgFrmSrv(int port, CmbImgCache* img_cache);
        ~CmbImgFrmSrv();

    };
}

#endif
