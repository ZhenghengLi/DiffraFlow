#include "CmbImgDataSrv.hh"
#include "CmbImgDataConn.hh"
#include "CmbImgCache.hh"

diffraflow::CmbImgDataSrv::CmbImgDataSrv(string host, int port,
    CmbImgCache* img_cache): GenericServer(host, port) {
    image_cache_ = img_cache;
}

diffraflow::CmbImgDataSrv::~CmbImgDataSrv() {

}

diffraflow::GenericConnection* diffraflow::CmbImgDataSrv::new_connection_(int client_sock_fd) {
    return new CmbImgDataConn(client_sock_fd, image_cache_);
}
