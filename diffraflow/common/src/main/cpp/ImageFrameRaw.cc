#include "ImageFrameRaw.hh"
#include "Decoder.hh"
#include <cstring>

#define FRAME_SIZE 131096

diffraflow::ImageFrameRaw::ImageFrameRaw() {
    module_id = -1;
    bunch_id = 0;
    data_buffer_ = nullptr;
    data_size_ = 0;
}

diffraflow::ImageFrameRaw::~ImageFrameRaw() {
    if (data_buffer_ != nullptr) {
        delete[] data_buffer_;
        data_buffer_ = nullptr;
    }
}

bool diffraflow::ImageFrameRaw::set_data(const char* buffer, const size_t len) {
    if (len != FRAME_SIZE) return false;
    if (data_buffer_ == nullptr) {
        data_buffer_ = new char[FRAME_SIZE];
    }
    bunch_id = gDC.decode_byte<uint64_t>(buffer, 12, 19);
    module_id = gDC.decode_byte<uint16_t>(buffer, 6, 7);
    memcpy(data_buffer_, buffer, len);
    data_size_ = len;
    return true;
}

uint64_t diffraflow::ImageFrameRaw::get_key() { return bunch_id; }

char* diffraflow::ImageFrameRaw::data() { return data_buffer_; }

size_t diffraflow::ImageFrameRaw::size() { return data_size_; }
