#ifndef CmbImgDatSrv_H
#define CmbImgDatSrv_H

#include "GenericServer.hh"

namespace diffraflow {

    class CmbImgDatConn;
    class CmbImgCache;

    class CmbImgDatSrv : public GenericServer {
    public:
        CmbImgDatSrv(string host, int port, CmbImgCache* img_cache);
        ~CmbImgDatSrv();

    protected:
        GenericConnection* new_connection_(int client_sock_fd);

    private:
        CmbImgCache* image_cache_;
    };
} // namespace diffraflow

#endif
