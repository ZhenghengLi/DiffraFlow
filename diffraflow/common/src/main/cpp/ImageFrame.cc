#include "ImageFrame.hh"

#include <algorithm>
#include <cassert>
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

#include "Decoder.hh"

using std::copy;
using std::cout;
using std::cerr;
using std::endl;

log4cxx::LoggerPtr diffraflow::ImageFrame::logger_
    = log4cxx::Logger::getLogger("ImageFrame");

diffraflow::ImageFrame::ImageFrame() {
    img_key = -1;
    img_time = -1;
    det_id = -1;
    img_width = -1;
    img_frame.clear();
    img_rawdata_.clear();
}

diffraflow::ImageFrame::~ImageFrame() {

}

bool diffraflow::ImageFrame::decode(const char* buffer, const size_t size) {
    if (size <= 8) return false;
    img_key = gDC.decode_byte<int64_t>(buffer, 0, 7);
    img_rawdata_.resize(size - 8);
    copy(buffer + 8, buffer + size, img_rawdata_.data());
    det_id = 0;
    return true;
}

void diffraflow::ImageFrame::print() const {
    if (img_rawdata_.empty()) {
        cout << "there is no data to print" << endl;
    }
    cout << "img_key: " << img_key << endl;
    cout << "img_data: [";
    for (size_t i = 0; i < img_rawdata_.size(); i++) {
        cout << img_rawdata_[i];
    }
    cout << "]" << endl;
}

bool diffraflow::ImageFrame::operator<(const diffraflow::ImageFrame& right) const {
    return img_time < right.img_time;
}

bool diffraflow::ImageFrame::operator<=(const diffraflow::ImageFrame& right) const {
    return img_time <= right.img_time;
}

bool diffraflow::ImageFrame::operator>(const diffraflow::ImageFrame& right) const {
    return img_time > right.img_time;
}

bool diffraflow::ImageFrame::operator>=(const diffraflow::ImageFrame& right) const {
    return img_time >= right.img_time;
}

bool diffraflow::ImageFrame::operator==(const diffraflow::ImageFrame& right) const {
    return img_time == right.img_time;
}

double diffraflow::ImageFrame::operator-(const diffraflow::ImageFrame& right) const {
    return img_time - right.img_time;
}
