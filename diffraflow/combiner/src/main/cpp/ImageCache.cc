#include "ImageCache.hh"
#include "ImageFrame.hh"
#include "ImageData.hh"

#include <iostream>

using std::cout;
using std::endl;
using diffraflow::ImageData;

diffraflow::ImageCache::ImageCache() {
    image_data_queue_.set_maxsize(100);
}

diffraflow::ImageCache::~ImageCache() {

}

void diffraflow::ImageCache::put_frame(ImageFrame& image_frame) {
    // in this function, put image from into priority queue, then try to do time alignment
    cout << "ImageCache::put_frame:" << endl;
    image_frame.print();
    ImageData image_data;
    image_data.put_imgfrm(0, image_frame);
    image_data_queue_.push(image_data);
}

bool diffraflow::ImageCache::take_one_image(ImageData& image_data, int timeout_ms) {
    return image_data_queue_.take(image_data, timeout_ms);
}

void diffraflow::ImageCache::img_queue_stop() {
    image_data_queue_.stop();
}

bool diffraflow::ImageCache::img_queue_stopped() {
    return image_data_queue_.stopped();
}
