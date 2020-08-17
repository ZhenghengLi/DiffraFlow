#include "ImageData.hh"
#include "ImageFrame.hh"
#include "Decoder.hh"

#include <iostream>
#include <algorithm>
#include <stdexcept>

#define FRAME_SIZE 131096

using std::endl;
using std::string;

diffraflow::ImageData::ImageData(uint32_t numOfMod) {
    bunch_id = 0;
    late_arrived = false;
    is_defined_ = false;
    calib_level_ = 0;
    if (numOfMod > 0) {
        alignment_vec.resize(numOfMod, false);
        image_frame_vec.resize(numOfMod);
    }
}

diffraflow::ImageData::~ImageData() {}

uint64_t diffraflow::ImageData::get_key() { return bunch_id; }

void diffraflow::ImageData::set_key(uint64_t key) { bunch_id = key; }

bool diffraflow::ImageData::put_imgfrm(size_t index, const shared_ptr<ImageFrame>& imgfrm) {
    if (index >= image_frame_vec.size()) return false;
    alignment_vec[index] = true;
    image_frame_vec[index] = imgfrm;
    return true;
}

void diffraflow::ImageData::set_defined() { is_defined_ = true; }

bool diffraflow::ImageData::get_defined() { return is_defined_; }

void diffraflow::ImageData::set_calib_level(int level) { calib_level_ = level; }

int diffraflow::ImageData::get_calib_level() { return calib_level_; }

bool diffraflow::ImageData::decode(const char* buffer, const size_t len) {
    if (len < 11) return false;
    bunch_id = gDC.decode_byte<uint64_t>(buffer, 0, 7);
    uint16_t alignment_bits = gDC.decode_byte<uint16_t>(buffer, 8, 9);
    alignment_vec.resize(16, false);
    image_frame_vec.resize(16);
    for (size_t i = 0; i < 16; i++) {
        alignment_vec[i] = (1 << (15 - i)) & alignment_bits;
    }
    late_arrived = gDC.decode_byte<uint8_t>(buffer, 10, 10);
    size_t current_pos = 11;
    for (size_t i = 0; i < 16; i++) {
        if (alignment_vec[i]) {
            if (len - current_pos < FRAME_SIZE) return false;
            image_frame_vec[i] = make_shared<ImageFrame>();
            image_frame_vec[i]->decode(buffer + current_pos, FRAME_SIZE);
            current_pos += FRAME_SIZE;
        }
    }
    is_defined_ = true;
    return true;
}

void diffraflow::ImageData::print(ostream& out) const {
    if (!is_defined_) {
        out << "undefined image data" << endl;
        return;
    }
    out << "bunch_id: " << bunch_id << endl;
    out << "late_arrived: " << late_arrived << endl;
    out << "alignment_vec: [";
    for (size_t i = 0; i < alignment_vec.size(); i++) {
        if (i > 0) out << ", ";
        out << alignment_vec[i];
    }
    out << "]" << endl;
    for (size_t i = 0; i < 16; i++) {
        if (alignment_vec[i]) {
            out << "image_frame_vec[" << i << "]:" << endl;
            image_frame_vec[i]->print(out);
            return;
        }
    }
}
