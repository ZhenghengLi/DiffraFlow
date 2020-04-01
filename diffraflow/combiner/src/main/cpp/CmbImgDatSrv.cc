#include "CmbImgDatSrv.hh"
#include "CmbImgDatConn.hh"
#include "CmbImgCache.hh"

diffraflow::CmbImgDatSrv::CmbImgDatSrv(string host, int port,
    CmbImgCache* img_cache): GenericServer(host, port) {
    image_cache_ = img_cache;
}

diffraflow::CmbImgDatSrv::~CmbImgDatSrv() {

}

diffraflow::GenericConnection* diffraflow::CmbImgDatSrv::new_connection_(int client_sock_fd) {
    return new CmbImgDatConn(client_sock_fd, image_cache_);
}
