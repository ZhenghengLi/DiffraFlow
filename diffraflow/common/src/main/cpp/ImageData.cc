#include "ImageData.hh"
#include "ImageFrame.hh"

#include <iostream>
#include <algorithm>
#include <stdexcept>

using std::endl;
using std::string;

diffraflow::ImageData::ImageData(uint32_t numOfDet) {
    event_time = 0;
    alignment_vec.resize(numOfDet, false);
    image_frame_vec.resize(numOfDet);
    late_arrived = false;
    wait_threshold = 0;
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
    out << "event_time: " << event_time << endl;
    out << "late_arrived: " << late_arrived << endl;
    out << "alignment_vec: [";
    for (size_t i = 0; i < alignment_vec.size(); i++) {
        if (i > 0) out << ", ";
        if (alignment_vec[i]) {
            out << "true";
        } else {
            out << "false";
        }
    }
    out << "]" << endl;
    out << "image_frame_vec:" << endl;
    for (size_t i = 0; i < image_frame_vec.size(); i++) {
        string rawdata_str(image_frame_vec[i].image_rawdata.data(), image_frame_vec[i].image_rawdata.size());
        out << "- [" << rawdata_str << "]" << endl;
    }
}
