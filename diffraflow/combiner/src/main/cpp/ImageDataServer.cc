#include "ImageDataServer.hh"
#include "ImageDataConnection.hh"
#include "ImageCache.hh"


using std::cout;
using std::cerr;
using std::endl;

diffraflow::ImageDataServer::ImageDataServer(
    string sock_path, ImageCache* img_cache): GeneralServer(sock_path) {
    image_cache_ = img_cache;
}

diffraflow::ImageDataServer::~ImageDataServer() {

}

diffraflow::GeneralConnection* diffraflow::ImageDataServer::new_connection_(int client_sock_fd) {
    return new ImageDataConnection(client_sock_fd, image_cache_);
}
