#ifndef CmbImgFrmSrv_H
#define CmbImgFrmSrv_H

#include "GenericServer.hh"

namespace diffraflow {

    class CmbImgFrmConn;
    class CmbImgCache;

    class CmbImgFrmSrv : public GenericServer {
    public:
        CmbImgFrmSrv(string host, int port, CmbImgCache* img_cache);
        ~CmbImgFrmSrv();

    protected:
        GenericConnection* new_connection_(int client_sock_fd);

    private:
        CmbImgCache* image_cache_;
    };
} // namespace diffraflow

#endif
