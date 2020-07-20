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
using std::flush;

log4cxx::LoggerPtr diffraflow::ImageFrame::logger_ = log4cxx::Logger::getLogger("ImageFrame");

diffraflow::ImageFrame::ImageFrame() {
    bunch_id = 0;
    module_id = -1;
    cell_id = -1;
    status = 0;

    gain_level.resize(65536);
    pixel_data.resize(65536);
}

diffraflow::ImageFrame::~ImageFrame() {}

uint64_t diffraflow::ImageFrame::get_key() { return bunch_id; }

bool diffraflow::ImageFrame::decode(const char* buffer, const size_t size) {
    if (size <= 131096) return false;

    bunch_id = gDC.decode_byte<uint64_t>(buffer, 12, 19);
    module_id = gDC.decode_byte<uint16_t>(buffer, 6, 7);
    if (module_id > 15) {
        LOG4CXX_WARN(logger_, "found wrong module_id " << module_id << " with bunch_id " << bunch_id);
        return false;
    }
    cell_id = gDC.decode_byte<uint16_t>(buffer, 8, 9);

    for (size_t i = 0; i < 65536; i++) {
        size_t offset = 20 + i * 2;
        gain_level[i] = gDC.decode_bit<uint8_t>(buffer + offset, 0, 1);
        pixel_data[i] = gDC.decode_bit<uint16_t>(buffer + offset, 2, 15);
    }

    return true;
}

void diffraflow::ImageFrame::print(ostream& out) const {
    out << "  bunch_id: " << bunch_id << endl;
    out << "  module_id: " << module_id << endl;
    out << "  cell_id: " << cell_id << endl;
    out << "  pixel_data: [";
    for (size_t i = 0; i < 5; i++) {
        out << pixel_data[i] << ", ";
    }
    out << "...";
    for (size_t i = 65531; i < 65536; i++) {
        out << ", " << pixel_data[i];
    }
    out << "]" << endl;
    out << " gain_level: [";
    for (size_t i = 0; i < 5; i++) {
        out << gain_level[i] << ", ";
    }
    out << "...";
    for (size_t i = 65531; i < 65536; i++) {
        out << ", " << gain_level[i];
    }
    out << "]" << endl;
}
