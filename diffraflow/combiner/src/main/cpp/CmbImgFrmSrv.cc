#include "CmbImgFrmSrv.hh"
#include "CmbImgFrmConn.hh"
#include "CmbImgCache.hh"

using std::cout;
using std::cerr;
using std::endl;

diffraflow::CmbImgFrmSrv::CmbImgFrmSrv(string host, int port,
    CmbImgCache* img_cache): GenericServer(host, port) {
    image_cache_ = img_cache;
}

diffraflow::CmbImgFrmSrv::~CmbImgFrmSrv() {

}

diffraflow::GenericConnection* diffraflow::CmbImgFrmSrv::new_connection_(int client_sock_fd) {
    return new CmbImgFrmConn(client_sock_fd, image_cache_);
}

Json::Value diffraflow::CmbImgFrmSrv::collect_metrics() {
    lock_guard<mutex> lk(mtx_);
    Json::Value connection_metrics_json;
    for (connListT_::iterator iter = connections_.begin(); iter != connections_.end();) {
        CmbImgFrmConn* current_connection = dynamic_cast<CmbImgFrmConn*>(iter->first);
        if (!current_connection->done()) {
            connection_metrics_json.append(
                current_connection->collect_metrics()
            );
        }
    }
}