#include "ImageFrameServer.hh"
#include "ImageFrameConnection.hh"
#include "ImageCache.hh"

using std::cout;
using std::cerr;
using std::endl;

shine::ImageFrameServer::ImageFrameServer(
    int port, ImageCache* img_cache): GeneralServer(port) {
    image_cache_ = img_cache;
}

shine::ImageFrameServer::~ImageFrameServer() {

}

shine::GeneralConnection* shine::ImageFrameServer::new_connection_(int client_sock_fd) {
    return new ImageFrameConnection(client_sock_fd, image_cache_);
}
