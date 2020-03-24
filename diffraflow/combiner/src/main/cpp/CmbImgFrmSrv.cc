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

json::value diffraflow::CmbImgFrmSrv::collect_metrics() {
    lock_guard<mutex> lk(mtx_);
    json::value connection_metrics_json;
    int array_index = 0;
    for (connListT_::iterator iter = connections_.begin(); iter != connections_.end(); ++iter) {
        CmbImgFrmConn* current_connection = dynamic_cast<CmbImgFrmConn*>(iter->first);
        if (!current_connection->done()) {
            connection_metrics_json[array_index++] = current_connection->collect_metrics();
        }
    }
    return connection_metrics_json;
}