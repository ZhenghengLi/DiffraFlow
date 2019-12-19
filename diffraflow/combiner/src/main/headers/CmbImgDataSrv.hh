#ifndef CmbImgDataSrv_H
#define CmbImgDataSrv_H

#include "GenericServer.hh"

namespace diffraflow {

    class CmbImgDataConn;
    class CmbImgCache;

    class CmbImgDataSrv: public GenericServer {
    private:
        CmbImgCache* image_cache_;

    protected:
        GenericConnection* new_connection_(int client_sock_fd);

    public:
        CmbImgDataSrv(string sock_path, CmbImgCache* img_cache);
        ~CmbImgDataSrv();

    };
}

#endif
