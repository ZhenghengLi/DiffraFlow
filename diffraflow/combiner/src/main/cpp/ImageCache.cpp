#include "ImageCache.hpp"
#include "ImageFrame.hpp"
#include "ImageData.hpp"

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