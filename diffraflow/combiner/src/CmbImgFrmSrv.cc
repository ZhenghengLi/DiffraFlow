#include "CmbImgFrmSrv.hh"
#include "CmbImgFrmConn.hh"
#include "CmbImgCache.hh"

using std::cout;
using std::cerr;
using std::endl;

diffraflow::CmbImgFrmSrv::CmbImgFrmSrv(string host, int port, CmbImgCache* img_cache) : GenericServer(host, port) {
    image_cache_ = img_cache;
}

diffraflow::CmbImgFrmSrv::~CmbImgFrmSrv() {}

diffraflow::GenericConnection* diffraflow::CmbImgFrmSrv::new_connection_(int client_sock_fd) {
    return new CmbImgFrmConn(client_sock_fd, image_cache_);
}
