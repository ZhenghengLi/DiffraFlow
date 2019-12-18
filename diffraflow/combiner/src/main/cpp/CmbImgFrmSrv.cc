#include "CmbImgFrmSrv.hh"
#include "CmbImgFrmConn.hh"
#include "CmbImgCache.hh"

using std::cout;
using std::cerr;
using std::endl;

diffraflow::CmbImgFrmSrv::CmbImgFrmSrv(
    int port, CmbImgCache* img_cache): GeneralServer(port) {
    image_cache_ = img_cache;
}

diffraflow::CmbImgFrmSrv::~CmbImgFrmSrv() {

}

diffraflow::GeneralConnection* diffraflow::CmbImgFrmSrv::new_connection_(int client_sock_fd) {
    return new CmbImgFrmConn(client_sock_fd, image_cache_);
}
