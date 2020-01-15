#ifndef CmbImgFrmSrv_H
#define CmbImgFrmSrv_H

#include "GenericServer.hh"

namespace diffraflow {

    class CmbImgFrmConn;
    class CmbImgCache;

    class CmbImgFrmSrv: public GenericServer {
    private:
        CmbImgCache* image_cache_;

    protected:
        GenericConnection* new_connection_(int client_sock_fd);

    public:
        CmbImgFrmSrv(string host, int port, CmbImgCache* img_cache);
        ~CmbImgFrmSrv();

    };
}

#endif
