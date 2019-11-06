#include "ImageFrame.hh"
#include <algorithm>
#include <cassert>

using std::copy;
using std::cout;
using std::cerr;
using std::endl;

shine::ImageFrame::ImageFrame() {
    img_frame = nullptr;
    img_rawdata_ = nullptr;
    img_rawsize_ = 0;
}

shine::ImageFrame::ImageFrame(const char* buffer, const size_t size) {
    img_frame = nullptr;
    img_rawdata_ = nullptr;
    img_rawsize_ = 0;
    decode(buffer, size);
}

shine::ImageFrame::ImageFrame(const ImageFrame& img_frm) {
    copyObj_(img_frm);
}

shine::ImageFrame::~ImageFrame() {
    if (img_frame != nullptr) delete [] img_frame;
    if (img_rawdata_ != nullptr) delete [] img_rawdata_;
}

void shine::ImageFrame::copyObj_(const ImageFrame& img_frm) {
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

shine::ImageFrame& shine::ImageFrame::operator=(const ImageFrame& img_frm) {
    copyObj_(img_frm);
    return *this;
}

bool shine::ImageFrame::decode(const char* buffer, const size_t size) {
    clear_data();
    if (size < 8) return false;
    img_key = gDC.decode_byte<int64_t>(buffer, 0, 7);
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

size_t shine::ImageFrame::serialize(char* const data, size_t len) {
    size_t gOffset = 0, offset = 0;
    // head
    offset = gPS.serializeValue<uint32_t>(kObjectHead, data + gOffset, len - gOffset);
    if (offset > 0) gOffset += offset; else return 0;
    // size
    offset = gPS.serializeValue<uint32_t>(object_size(), data + gOffset, len - gOffset);
    if (offset > 0) gOffset += offset; else return 0;
    // type
    offset = gPS.serializeValue<int32_t>(object_type(), data + gOffset, len - gOffset);
    if (offset > 0) gOffset += offset; else return 0;
    // data
    // - img_rawsize_
    offset = gPS.serializeValue<uint32_t>(img_rawsize_, data + gOffset, len - gOffset);
    if (offset > 0) gOffset += offset; else return 0;
    // - img_rawdata_
    if (img_rawsize_ > len - gOffset) return 0;
    for (size_t i = 0; i < img_rawsize_; i++) {
        (data + gOffset)[i] = img_rawdata_[i];
    }
    gOffset += img_rawsize_;
    return gOffset;
}

size_t shine::ImageFrame::deserialize(const char* const data, size_t len) {
    clear_data();
    size_t gOffset = 0, offset = 0;
    // check type
    int objType = 0;
    offset = gPS.deserializeValue<int32_t>(&objType, data + gOffset, len - gOffset);
    if (offset > 0) gOffset += offset; else return 0;
    if (objType != object_type()) {
        cerr << "WARNING: the type of object to deserialize does not match." << endl;
        return 0;
    }
    // read data
    // - img_rawsize_
    offset = gPS.deserializeValue<uint32_t>(&img_rawsize_, data + gOffset, len - gOffset);
    if (offset > 0) gOffset += offset; else return 0;
    // - img_rawdata_
    if (img_rawsize_ > 0) {
        img_rawdata_ = new char[img_rawsize_];
        for (size_t i = 0; i < img_rawsize_; i++) {
            img_rawdata_[i] = (data + gOffset)[i];
        }
    }
    gOffset += img_rawsize_;
    return gOffset;
}

size_t shine::ImageFrame::object_size() {
    size_t theSize = 0;
    // type
    theSize += sizeof(int32_t);
    // data
    // - img_rawsize_
    theSize += sizeof(uint32_t);
    // - img_rawdata_
    theSize += img_rawsize_;
    return theSize;
}

int shine::ImageFrame::object_type() {
    return 1231;
}

void shine::ImageFrame::clear_data() {
    img_rawsize_ = 0;
    if (img_rawdata_ != nullptr) {
        delete [] img_rawdata_;
    }
    img_rawdata_ = nullptr;
    if (img_frame != nullptr) {
        delete [] img_frame;
    }
    img_frame = nullptr;
}
