#include "ImageFrame.hpp"
#include <algorithm>

using std::copy;
using std::cout;
using std::endl;

shine::ImageFrame::ImageFrame() {
    img_frame = nullptr;
    img_rawdata_ = nullptr;
}

shine::ImageFrame::ImageFrame(const ImageFrame& img_frm) {
    det_id = img_frm.det_id;
    img_width = img_frm.img_width;
    img_height = img_frm.img_height;
    img_key = img_frm.img_key;
    if (img_frm.img_frame != nullptr) {
        size_t size = img_width * img_height;
        img_frame = new float[size];
        copy(img_frm.img_frame, img_frm.img_frame + size, img_frame);
    } else {
        img_frame = nullptr;
    }
    img_rawsize_ = img_frm.img_rawsize_;
    if (img_frm.img_rawdata_ != nullptr) {
        img_rawdata_ = new char[img_rawsize_];
        copy(img_frm.img_rawdata_, img_frm.img_rawdata_ + img_rawsize_, img_rawdata_);
    } else {
        img_rawdata_ = nullptr;
    }
}

shine::ImageFrame::~ImageFrame() {

}

bool shine::ImageFrame::decode(const char* img_rd, const size_t img_sz) {
    if (img_rawdata_ != nullptr) {
        delete [] img_rawdata_;
    }
    img_rawdata_ = new char[img_sz];
    copy(img_rd, img_rd + img_sz, img_rawdata_);
    img_rawsize_ = img_sz;
    return true;
}

void shine::ImageFrame::print() {
    if (img_rawdata_ == nullptr) {
        cout << "there is no data to print" << endl;
    }
    cout << "img_key: " << img_key << endl;
    cout << "img_data: [";
    for (size_t i = 0; i < img_rawsize_; i++) {
        cout << img_rawdata_[i];
    }
    cout << "]" << endl;
}
