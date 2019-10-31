#include "ImageCache.hh"
#include "ImageFrame.hh"
#include "ImageData.hh"

using shine::ImageData;

shine::ImageCache::ImageCache() {

}

shine::ImageCache::~ImageCache() {

}

void shine::ImageCache::put_frame(ImageFrame& image_frame) {
    image_frame.print();

}

ImageData shine::ImageCache::take_one_image() {

}