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
    imgFrm_len = numOfDet;
    imgFrm_arr = new ImageFrame[imgFrm_len];
    status_arr = new uint8_t[imgFrm_len];
    for (size_t i = 0; i < imgFrm_len; i++) {
        status_arr[i] = 0x0;
    }
}

diffraflow::ImageData::ImageData(const ImageData& img_data) {
    copyObj_(img_data);
}

diffraflow::ImageData& diffraflow::ImageData::operator=(const ImageData& img_data) {
    copyObj_(img_data);
}

void diffraflow::ImageData::copyObj_(const ImageData& img_data) {
    if (imgFrm_len != img_data.imgFrm_len) {
        imgFrm_len = img_data.imgFrm_len;
        delete [] imgFrm_arr;
        imgFrm_arr = new ImageFrame[imgFrm_len];
        delete [] status_arr;
        status_arr = new uint8_t[imgFrm_len];
    }
    copy(img_data.imgFrm_arr, img_data.imgFrm_arr + imgFrm_len, imgFrm_arr);
    copy(img_data.status_arr, img_data.status_arr + imgFrm_len, status_arr);
    event_key = img_data.event_key;
    event_time = img_data.event_time;
}

diffraflow::ImageData::~ImageData() {
    delete [] imgFrm_arr;
    delete [] status_arr;
}

bool diffraflow::ImageData::put_imgfrm(size_t index, const ImageFrame& imgfrm) {
    if (index >= imgFrm_len) return false;
    imgFrm_arr[index] = imgfrm;
    status_arr[index] = 0x1;
    return true;
}

void diffraflow::ImageData::print() {
    cout << "ImageData[0]:" << endl;
    imgFrm_arr[0].print();
}

size_t diffraflow::ImageData::serialize(char* const data, size_t len) {
    // note: this function may throw exceptions
    size_t offset = 0;
    // type
    offset += gPS.serializeValue<int32_t>(object_type(), data + offset, len - offset);
    // data
    // - event_key
    offset += gPS.serializeValue<int64_t>(event_key, data + offset, len - offset);
    // - event_time
    offset += gPS.serializeValue<double>(event_time, data + offset, len - offset);
    // - imgFrm_len
    offset += gPS.serializeValue<uint32_t>(imgFrm_len, data + offset, len - offset);
    // - status_arr
    for (size_t i = 0; i < imgFrm_len; i++) {
        offset += gPS.serializeValue<uint8_t>(status_arr[i], data + offset, len - offset);
    }
    // - imgFrm_arr
    for (size_t i = 0; i < imgFrm_len; i++) {
        if (status_arr[i] == 0x0) continue;  // only serialize valid image frame.
        offset += imgFrm_arr[i].serialize(data + offset, len - offset);
    }
    // return
    return offset;
}

size_t diffraflow::ImageData::deserialize(const char* const data, size_t len) {
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
    // - event_key
    offset += gPS.deserializeValue<int64_t>(&event_key, data + offset, len - offset);
    // -event_time
    offset += gPS.deserializeValue<double>(&event_time, data + offset, len - offset);
    // - imgFrm_len
    offset += gPS.deserializeValue<uint32_t>(&imgFrm_len, data + offset, len - offset);
    // - status_arr
    for (size_t i = 0; i < imgFrm_len; i++) {
        offset += gPS.deserializeValue<uint8_t>(&status_arr[i], data + offset, len - offset);
    }
    // - imgFrm_arr
    for (size_t i = 0; i < imgFrm_len; i++) {
        if (status_arr[i] == 0x0) continue;  // skip invalid image frame
        offset += imgFrm_arr[i].deserialize(data + offset, len - offset);
    }
    // return
    return offset;
}

size_t diffraflow::ImageData::object_size() {
    size_t theSize = 0;
    // type
    theSize += sizeof(int32_t);
    // data
    // - event_key
    theSize += sizeof(int64_t);
    // - event_time
    theSize += sizeof(double);
    // - imgFrm_len
    theSize += sizeof(uint32_t);
    // - status_arr
    theSize += sizeof(uint8_t) * imgFrm_len;
    // - imgFrm_arr
    for (size_t i = 0; i < imgFrm_len; i++) {
        if (status_arr[i] == 0x0) continue;
        theSize += imgFrm_arr[i].object_size();
    }
    // return
    return theSize;
}

int diffraflow::ImageData::object_type() {
    return obj_type_;
}

void diffraflow::ImageData::clear_data() {
    for (size_t i = 0; i < imgFrm_len; i++) {
        status_arr[i] = 0x0;
    }
}
