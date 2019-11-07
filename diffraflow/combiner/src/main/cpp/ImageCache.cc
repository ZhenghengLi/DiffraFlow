#include "ImageCache.hh"
#include "ImageFrame.hh"
#include "ImageData.hh"

#include <iostream>

using std::cout;
using std::endl;
using shine::ImageData;

shine::ImageCache::ImageCache() {
    image_data_queue_.set_maxsize(100);
}

shine::ImageCache::~ImageCache() {

}

void shine::ImageCache::put_frame(ImageFrame& image_frame) {
    // in this function, put image from into priority queue, then try to do time alignment
    cout << "ImageCache::put_frame:" << endl;
    image_frame.print();
    ImageData image_data;
    image_data.put_imgfrm(0, image_frame);
    image_data_queue_.push(image_data);
}

bool shine::ImageCache::take_one_image(ImageData& image_data) {
    return image_data_queue_.take(image_data);
}