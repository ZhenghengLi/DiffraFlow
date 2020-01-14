#include "ImageFrame.hh"
#include <algorithm>
#include <cassert>
#include <stdexcept>

using std::copy;
using std::cout;
using std::cerr;
using std::endl;

diffraflow::ImageFrame::ImageFrame() {
    img_frame = nullptr;
    img_rawdata_ = nullptr;
    img_rawsize_ = 0;
    img_width = 0;
    img_height = 0;
}

diffraflow::ImageFrame::ImageFrame(const char* buffer, const size_t size) {
    img_frame = nullptr;
    img_rawdata_ = nullptr;
    img_rawsize_ = 0;
    img_width = 0;
    img_height = 0;
    decode(buffer, size);
}

diffraflow::ImageFrame::ImageFrame(const ImageFrame& img_frm) {
    copyObj_(img_frm);
}

diffraflow::ImageFrame::~ImageFrame() {
    if (img_frame != nullptr) delete [] img_frame;
    if (img_rawdata_ != nullptr) delete [] img_rawdata_;
}

void diffraflow::ImageFrame::copyObj_(const ImageFrame& img_frm) {
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

diffraflow::ImageFrame& diffraflow::ImageFrame::operator=(const ImageFrame& img_frm) {
    copyObj_(img_frm);
    return *this;
}

bool diffraflow::ImageFrame::decode(const char* buffer, const size_t size) {
    clear_data();
    if (size < 8) return false;
    img_key = gDC.decode_byte<int64_t>(buffer, 0, 7);
    assert(size > 8);
    img_rawdata_ = new char[size - 8];
    copy(buffer + 8, buffer + size, img_rawdata_);
    img_rawsize_ = size - 8;
    det_id = 0;
    return true;
}

void diffraflow::ImageFrame::print() const {
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

size_t diffraflow::ImageFrame::serialize(char* const data, size_t len) {
    // note: this function may throw exceptions
    size_t offset = 0;
    // type
    offset += gPS.serializeValue<int32_t>(object_type(), data + offset, len - offset);
    // data
    // - img_rawsize_
    offset += gPS.serializeValue<uint32_t>(img_rawsize_, data + offset, len - offset);
    // - img_rawdata_
    for (size_t i = 0; i < img_rawsize_; i++) {
        offset += gPS.serializeValue<char>(img_rawdata_[i], data + offset, len - offset);
    }
    // - img_key
    offset += gPS.serializeValue<int64_t>(img_key, data + offset, len - offset);
    // - img_time
    offset += gPS.serializeValue<double>(img_time, data + offset, len - offset);
    // - det_id
    offset += gPS.serializeValue<uint32_t>(det_id, data + offset, len - offset);
    // - img_width
    offset += gPS.serializeValue<uint32_t>(img_width, data + offset, len - offset);
    // - img_height
    offset += gPS.serializeValue<uint32_t>(img_height, data + offset, len - offset);
    // - img_frame
    size_t frm_size = img_width * img_height;
    for (size_t i = 0; i < frm_size; i++) {
        offset += gPS.serializeValue<float>(img_frame[i], data + offset, len - offset);
    }
    // return
    return offset;
}

size_t diffraflow::ImageFrame::deserialize(const char* const data, size_t len) {
    // note: this function may throw exceptions
    clear_data();
    size_t offset = 0;
    // check type
    int objType = 0;
    offset += gPS.deserializeValue<int32_t>(&objType, data + offset, len - offset);
    if (objType != object_type()) {
        throw std::invalid_argument("the type of object to deserialize does not match.");
    }
    // read data
    // - img_rawsize_
    offset += gPS.deserializeValue<uint32_t>(&img_rawsize_, data + offset, len - offset);
    // - img_rawdata_
    if (img_rawsize_ > 0) {
        img_rawdata_ = new char[img_rawsize_];
        for (size_t i = 0; i < img_rawsize_; i++) {
            offset += gPS.deserializeValue<char>(&img_rawdata_[i], data + offset, len - offset);
        }
    }
    // - img_key
    offset += gPS.deserializeValue<int64_t>(&img_key, data + offset, len - offset);
    // - img_time
    offset += gPS.deserializeValue<double>(&img_time, data + offset, len - offset);
    // - det_id
    offset += gPS.deserializeValue<uint32_t>(&det_id, data + offset, len - offset);
    // img_width
    offset += gPS.deserializeValue<uint32_t>(&img_width, data + offset, len - offset);
    // img_height
    offset += gPS.deserializeValue<uint32_t>(&img_height, data + offset, len - offset);
    // img_frame
    size_t frm_size = img_width * img_height;
    if (frm_size> 0) {
        img_frame = new float[frm_size];
        for (size_t i = 0; i < frm_size; i++) {
            offset += gPS.deserializeValue<float>(&img_frame[i], data + offset, len - offset);
        }
    }
    // return
    return offset;
}

size_t diffraflow::ImageFrame::object_size() {
    size_t theSize = 0;
    // type
    theSize += sizeof(int32_t);
    // data
    // - img_rawsize_
    theSize += sizeof(uint32_t);
    // - img_rawdata_
    theSize += img_rawsize_;
    // - img_key
    theSize += sizeof(int64_t);
    // - img_time
    theSize += sizeof(double);
    // - det_id
    theSize += sizeof(uint32_t);
    // - img_width
    theSize += sizeof(uint32_t);
    // - img_height
    theSize += sizeof(uint32_t);
    // - img_frame
    theSize += img_width * img_height * sizeof(float);
    // return
    return theSize;
}

int diffraflow::ImageFrame::object_type() {
    return obj_type_;
}

void diffraflow::ImageFrame::clear_data() {
    img_rawsize_ = 0;
    if (img_rawdata_ != nullptr) {
        delete [] img_rawdata_;
    }
    img_rawdata_ = nullptr;
    img_width = 0;
    img_height = 0;
    if (img_frame != nullptr) {
        delete [] img_frame;
    }
    img_frame = nullptr;
}
