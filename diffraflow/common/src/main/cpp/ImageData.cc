#include "ImageData.hh"
#include "ImageFrame.hh"

#include <iostream>
#include <algorithm>
#include <stdexcept>

using std::cout;
using std::endl;
using std::cerr;
using std::copy;

diffraflow::ImageData::ImageData(uint32_t numOfDet) {
    event_time = 0;
    alignment_vec.resize(numOfDet, false);
    image_frame_vec.resize(numOfDet);
    late_arrived = false;
    wait_threshold = 0;
    is_defined_ = false;
    calib_level_ = 0;
}

diffraflow::ImageData::~ImageData() {

}

bool diffraflow::ImageData::put_imgfrm(size_t index, const ImageFrame& imgfrm) {
    if (index >= image_frame_vec.size()) return false;
    alignment_vec[index] = true;
    image_frame_vec[index] = imgfrm;
    return true;
}

void diffraflow::ImageData::set_defined() {
    is_defined_ = true;
}

bool diffraflow::ImageData::get_defined() {
    return is_defined_;
}

void diffraflow::ImageData::set_calib_level(int level) {
    calib_level_ = level;
}

int diffraflow::ImageData::get_calib_level() {
    return calib_level_;
}

void diffraflow::ImageData::print() {
    for (size_t i = 0; i < image_frame_vec.size(); i++) {
        cout << "ImageData[" << i << "]:" << endl;
        image_frame_vec[i].print();
    }
    cout << "late_arrived: " << late_arrived << endl;
    cout << "is_defined: " << is_defined_ << endl;
    cout << "calib_level: " << calib_level_ << endl;
}
