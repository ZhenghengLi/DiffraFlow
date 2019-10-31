#include "ImageFrame.hh"
#include <algorithm>
#include <cassert>

using std::copy;
using std::cout;
using std::endl;

shine::ImageFrame::ImageFrame() {
    img_frame = nullptr;
    img_rawdata_ = nullptr;
}

shine::ImageFrame::ImageFrame(const char* buffer, const size_t size) {
    img_frame = nullptr;
    img_rawdata_ = nullptr;
    decode(buffer, size);
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
    if (img_frame != nullptr) delete [] img_frame;
    if (img_rawdata_ != nullptr) delete [] img_rawdata_;
}

bool shine::ImageFrame::decode(const char* buffer, const size_t size) {
    if (img_rawdata_ != nullptr) {
        delete [] img_rawdata_;
    }
    if (size < 8) return false;
    img_key = decode_byte<int64_t>(buffer, 0, 7);
    assert(size > 8);
    img_rawdata_ = new char[size - 8];
    copy(buffer + 8, buffer + size, img_rawdata_);
    img_rawsize_ = size - 8;
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
