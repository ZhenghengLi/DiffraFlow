#include "ImageFrameRaw.hh"
#include "Decoder.hh"
#include <cstring>

#define FRAME_SIZE 131096

diffraflow::ImageFrameRaw::ImageFrameRaw() {
    module_id = -1;
    bunch_id = 0;
}

diffraflow::ImageFrameRaw::~ImageFrameRaw() {}

bool diffraflow::ImageFrameRaw::set_data(const shared_ptr<vector<char>>& buffer) {
    if (!buffer) return false;
    if (buffer->size() != FRAME_SIZE) return false;

    bunch_id = gDC.decode_byte<uint64_t>(buffer->data(), 12, 19);
    module_id = gDC.decode_byte<uint16_t>(buffer->data(), 6, 7);
    data_buffer_ = buffer;

    return true;
}

uint64_t diffraflow::ImageFrameRaw::get_key() { return bunch_id; }

char* diffraflow::ImageFrameRaw::data() {
    if (data_buffer_) {
        return data_buffer_->data();
    } else {
        return nullptr;
    }
}

size_t diffraflow::ImageFrameRaw::size() {
    if (data_buffer_) {
        return data_buffer_->size();
    } else {
        return 0;
    }
}

bool diffraflow::ImageFrameRaw::add_dgram(shared_ptr<vector<char>>& dgram) {
    if (dgram->size() < 24) {
        return false;
    }
    if (dgram_list_.empty()) {
        dgram_mod_id = gDC.decode_byte<uint8_t>(dgram->data(), 0, 0);
        dgram_frm_sn = gDC.decode_byte<uint16_t>(dgram->data(), 1, 2);
        dgram_seg_sn = gDC.decode_byte<uint8_t>(dgram->data(), 3, 3);
    }
    dgram_list_.push_back(dgram);
    return true;
}

int diffraflow::ImageFrameRaw::check_dgrams_integrity() {
    //
    return -1;
}

void diffraflow::ImageFrameRaw::sort_dgrams() {
    //
}

shared_ptr<vector<char>>& diffraflow::ImageFrameRaw::get_dgram(size_t index) { return dgram_list_[index]; }

size_t diffraflow::ImageFrameRaw::get_dgram_count() { return dgram_list_.size(); }
