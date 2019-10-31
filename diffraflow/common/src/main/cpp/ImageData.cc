#include "ImageData.hh"
#include "ImageFrame.hh"

#include <iostream>

using std::cout;
using std::endl;

shine::ImageData::ImageData() {
    imgFrm_len_ = 1;
    imgFrm_arr_ = new ImageFrame*[imgFrm_len_];
    for (size_t i = 0; i < imgFrm_len_; i++) {
        imgFrm_arr_[i] = nullptr;
    }
}

shine::ImageData::~ImageData() {
    for (size_t i = 0; i < imgFrm_len_; i++) {
        if (imgFrm_arr_[i] != nullptr) {
            delete imgFrm_arr_[i];
        }
    }
    delete [] imgFrm_arr_;
}

bool shine::ImageData::put_imgfrm(size_t index, const ImageFrame& imgfrm) {
    if (index >= imgFrm_len_) return false;
    imgFrm_arr_[index] = new ImageFrame(imgfrm);
    return true;
}

bool shine::ImageData::serialize(char* data, size_t len) {
    cout << "do serialization" << endl;
    return true;
}

bool shine::ImageData::deserialize(char* data, size_t len) {

    return true;
}
