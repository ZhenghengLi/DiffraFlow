#include "ImageDataServer.hh"
#include "ImageCache.hh"
#include <cassert>

shine::ImageDataServer::ImageDataServer(ImageCache* img_cache) {
    assert(img_cache != nullptr);
    image_cache_ = img_cache;
}

shine::ImageDataServer::~ImageDataServer() {

}