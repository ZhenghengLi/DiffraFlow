#include "ImageData.hh"
#include "ImageFrame.hh"

#include <iostream>
#include <algorithm>
#include <stdexcept>

using std::endl;
using std::string;

diffraflow::ImageData::ImageData(uint32_t numOfMod) {
    bunch_id = 0;
    alignment_vec.resize(numOfMod, false);
    image_frame_vec.resize(numOfMod);
    late_arrived = false;
    is_defined_ = false;
    calib_level_ = 0;
}

diffraflow::ImageData::~ImageData() {}

bool diffraflow::ImageData::put_imgfrm(size_t index, const ImageFrame& imgfrm) {
    if (index >= image_frame_vec.size()) return false;
    alignment_vec[index] = true;
    image_frame_vec[index] = imgfrm;
    return true;
}

void diffraflow::ImageData::set_defined() { is_defined_ = true; }

bool diffraflow::ImageData::get_defined() { return is_defined_; }

void diffraflow::ImageData::set_calib_level(int level) { calib_level_ = level; }

int diffraflow::ImageData::get_calib_level() { return calib_level_; }

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
    if (image_frame_vec.size() > 0) {
        out << "image_frame_vec[0]:" << endl;
        image_frame_vec[0].print(out);
    }
}
