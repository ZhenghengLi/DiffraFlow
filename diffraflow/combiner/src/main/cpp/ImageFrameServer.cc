#include "ImageFrameServer.hh"
#include "ImageFrameConnection.hh"
#include "ImageCache.hh"

using std::cout;
using std::cerr;
using std::endl;

diffraflow::ImageFrameServer::ImageFrameServer(
    int port, ImageCache* img_cache): GeneralServer(port) {
    image_cache_ = img_cache;
}

diffraflow::ImageFrameServer::~ImageFrameServer() {

}

diffraflow::GeneralConnection* diffraflow::ImageFrameServer::new_connection_(int client_sock_fd) {
    return new ImageFrameConnection(client_sock_fd, image_cache_);
}
