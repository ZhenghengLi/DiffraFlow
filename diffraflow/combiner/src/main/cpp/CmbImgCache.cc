#include "CmbImgCache.hh"
#include "ImageFrame.hh"
#include "ImageData.hh"

#include <iostream>

using std::cout;
using std::endl;
using diffraflow::ImageData;

diffraflow::CmbImgCache::CmbImgCache() {
    image_data_queue_.set_maxsize(100);
}

diffraflow::CmbImgCache::~CmbImgCache() {

}

void diffraflow::CmbImgCache::put_frame(ImageFrame& image_frame) {
    // in this function, put image from into priority queue, then try to do time alignment
    cout << "CmbImgCache::put_frame:" << endl;
    image_frame.print();
    ImageData image_data;
    image_data.put_imgfrm(0, image_frame);
    image_data_queue_.push(image_data);
}

bool diffraflow::CmbImgCache::take_one_image(ImageData& image_data, int timeout_ms) {
    return image_data_queue_.take(image_data, timeout_ms);
}

void diffraflow::CmbImgCache::img_queue_stop() {
    image_data_queue_.stop();
}

bool diffraflow::CmbImgCache::img_queue_stopped() {
    return image_data_queue_.stopped();
}
