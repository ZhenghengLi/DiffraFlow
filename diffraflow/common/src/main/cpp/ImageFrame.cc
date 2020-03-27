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
    image_time = 0;
    detector_id = -1;
    image_width = 0;
    image_height = 0;
    image_frame.clear();
    image_rawdata_.clear();
}

diffraflow::ImageFrame::~ImageFrame() {

}

bool diffraflow::ImageFrame::decode(const char* buffer, const size_t size) {
    if (size <= 8) return false;
    image_time = gDC.decode_byte<uint64_t>(buffer, 0, 7);
    image_rawdata_.resize(size - 8);
    copy(buffer + 8, buffer + size, image_rawdata_.data());
    detector_id = 0;
    return true;
}

void diffraflow::ImageFrame::print() const {
    if (image_rawdata_.empty()) {
        cout << "there is no data to print" << endl;
    }
    cout << "image_time: " << image_time << endl;
    cout << "image_data: [";
    for (size_t i = 0; i < image_rawdata_.size(); i++) {
        cout << image_rawdata_[i];
    }
    cout << "]" << endl;
}

bool diffraflow::ImageFrame::operator<(const diffraflow::ImageFrame& right) const {
    return image_time < right.image_time;
}

bool diffraflow::ImageFrame::operator<=(const diffraflow::ImageFrame& right) const {
    return image_time <= right.image_time;
}

bool diffraflow::ImageFrame::operator>(const diffraflow::ImageFrame& right) const {
    return image_time > right.image_time;
}

bool diffraflow::ImageFrame::operator>=(const diffraflow::ImageFrame& right) const {
    return image_time >= right.image_time;
}

bool diffraflow::ImageFrame::operator==(const diffraflow::ImageFrame& right) const {
    return image_time == right.image_time;
}

int64_t diffraflow::ImageFrame::operator-(const diffraflow::ImageFrame& right) const {
    return image_time - right.image_time;
}
