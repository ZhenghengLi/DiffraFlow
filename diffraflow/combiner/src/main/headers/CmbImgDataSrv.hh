#ifndef CmbImgDataSrv_H
#define CmbImgDataSrv_H

#include "GeneralServer.hh"

namespace diffraflow {

    class CmbImgDataConn;
    class CmbImgCache;

    class CmbImgDataSrv: public GeneralServer {
    private:
        CmbImgCache* image_cache_;

    protected:
        GeneralConnection* new_connection_(int client_sock_fd);

    public:
        CmbImgDataSrv(string sock_path, CmbImgCache* img_cache);
        ~CmbImgDataSrv();

    };
}

#endif
