#include "ImageData.hh"
#include "ImageFrame.hh"

#include <iostream>
#include <algorithm>

using std::cout;
using std::endl;
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