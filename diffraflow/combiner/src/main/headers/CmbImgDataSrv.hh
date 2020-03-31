#ifndef CmbImgDataSrv_H
#define CmbImgDataSrv_H

#include "GenericServer.hh"

namespace diffraflow {

    class CmbImgDataConn;
    class CmbImgCache;

    class CmbImgDataSrv: public GenericServer {
    public:
        CmbImgDataSrv(string host, int port, CmbImgCache* img_cache);
        ~CmbImgDataSrv();

    protected:
        GenericConnection* new_connection_(int client_sock_fd);

    private:
        CmbImgCache* image_cache_;
    };
}

#endif
