#include "ImageData.hh"
#include "ImageFrame.hh"

#include <iostream>
#include <algorithm>

using std::cout;
using std::endl;
using std::cerr;
using std::copy;

shine::ImageData::ImageData() {
    ImageData(1);
}

shine::ImageData::ImageData(uint32_t numOfDet) {
    imgFrm_len = numOfDet;
    imgFrm_arr = new ImageFrame[imgFrm_len];
    status_arr = new uint8_t[imgFrm_len];
    for (size_t i = 0; i < imgFrm_len; i++) {
        status_arr[i] = 0x0;
    }
}

shine::ImageData::ImageData(const ImageData& img_data) {
    copyObj_(img_data);
}

shine::ImageData& shine::ImageData::operator=(const ImageData& img_data) {
    copyObj_(img_data);
}

void shine::ImageData::copyObj_(const ImageData& img_data) {
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

shine::ImageData::~ImageData() {
    delete [] imgFrm_arr;
    delete [] status_arr;
}

bool shine::ImageData::put_imgfrm(size_t index, const ImageFrame& imgfrm) {
    if (index >= imgFrm_len) return false;
    imgFrm_arr[index] = imgfrm;
    status_arr[index] = 0x1;
    return true;
}

size_t shine::ImageData::serialize(char* const data, size_t len) {
    size_t gOffset = 0, offset = 0;
    // type
    offset = gPS.serializeValue<int32_t>(object_type(), data + gOffset, len - gOffset);
    if (offset > 0) gOffset += offset; else return 0;
    // data
    // - event_key
    offset = gPS.serializeValue<int64_t>(event_key, data + gOffset, len - gOffset);
    if (offset > 0) gOffset += offset; else return 0;
    // - event_time
    offset = gPS.serializeValue<double>(event_time, data + gOffset, len - gOffset);
    if (offset > 0) gOffset += offset; else return 0;
    // - imgFrm_len
    offset = gPS.serializeValue<uint32_t>(imgFrm_len, data + gOffset, len - gOffset);
    if (offset > 0) gOffset += offset; else return 0;
    // - status_arr
    for (size_t i = 0; i < imgFrm_len; i++) {
        offset = gPS.serializeValue<uint8_t>(status_arr[i], data + gOffset, len - gOffset);
        if (offset > 0) gOffset += offset; else return 0;
    }
    // - imgFrm_arr
    for (size_t i = 0; i < imgFrm_len; i++) {
        if (status_arr[i] == 0x0) continue;  // only serialize valid image frame.
        offset = imgFrm_arr[i].serialize(data + gOffset, len - gOffset);
        if (offset > 0) gOffset += offset; else return 0;
    }
    return gOffset;
}

size_t shine::ImageData::deserialize(const char* const data, size_t len) {
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
    // - event_key
    offset = gPS.deserializeValue<int64_t>(&event_key, data + gOffset, len - gOffset);
    if (offset > 0) gOffset += offset; else return 0;
    // -event_time
    offset = gPS.deserializeValue<double>(&event_time, data + gOffset, len - gOffset);
    if (offset > 0) gOffset += offset; else return 0;
    // - imgFrm_len
    offset = gPS.deserializeValue<uint32_t>(&imgFrm_len, data + gOffset, len - gOffset);
    // - status_arr
    for (size_t i = 0; i < imgFrm_len; i++) {
        offset = gPS.deserializeValue<uint8_t>(&status_arr[i], data + gOffset, len - gOffset);
        if (offset > 0) gOffset += offset; else return 0;
    }
    // - imgFrm_arr
    for (size_t i = 0; i < imgFrm_len; i++) {
        if (status_arr[i] == 0x0) continue;  // skip invalid image frame
        offset = imgFrm_arr[i].deserialize(data + gOffset, len - gOffset);
        if (offset > 0) gOffset += offset; else return 0;
    }
    return gOffset;
}

size_t shine::ImageData::object_size() {
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
    return theSize;
}

int shine::ImageData::object_type() {
    return obj_type_;
}

void shine::ImageData::clear_data() {
    for (size_t i = 0; i < imgFrm_len; i++) {
        status_arr[i] = 0x0;
    }
}