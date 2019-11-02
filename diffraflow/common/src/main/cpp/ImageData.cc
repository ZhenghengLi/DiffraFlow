#include "ImageData.hh"
#include "ImageFrame.hh"

#include <iostream>

using std::cout;
using std::endl;

shine::ImageData::ImageData() {
    imgFrm_len_ = 1;
    imgFrm_arr_ = new ImageFrame[imgFrm_len_];
}

shine::ImageData::~ImageData() {
    delete [] imgFrm_arr_;
}

bool shine::ImageData::put_imgfrm(size_t index, const ImageFrame& imgfrm) {
    if (index >= imgFrm_len_) return false;
    imgFrm_arr_[index] = imgfrm;
    return true;
}

size_t shine::ImageData::serialize(char* const data, size_t len) {
    cout << "do serialization" << endl;
    return 0;
}

size_t shine::ImageData::deserialize(const char* const data, size_t len) {
    cout << "do deserialization" << endl;
    return 0;
}

size_t shine::ImageData::object_size() {
    return 0;
}

int shine::ImageData::object_type() {
    return 1232;
}

void shine::ImageData::clear_data() {

}