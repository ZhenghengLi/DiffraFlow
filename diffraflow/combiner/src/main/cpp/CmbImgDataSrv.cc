#include "CmbImgDataSrv.hh"
#include "CmbImgDataConn.hh"
#include "CmbImgCache.hh"


using std::cout;
using std::cerr;
using std::endl;

diffraflow::CmbImgDataSrv::CmbImgDataSrv(
    string sock_path, CmbImgCache* img_cache): GenericServer(sock_path) {
    image_cache_ = img_cache;
}

diffraflow::CmbImgDataSrv::~CmbImgDataSrv() {

}

diffraflow::GenericConnection* diffraflow::CmbImgDataSrv::new_connection_(int client_sock_fd) {
    return new CmbImgDataConn(client_sock_fd, image_cache_);
}
