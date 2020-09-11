#include "ImageFrameRaw.hh"
#include "Decoder.hh"
#include <cstring>
#include <boost/crc.hpp>

#define FRAME_SIZE 131096

using boost::crc_32_type;

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
    // (1) check dgram count
    if (dgram_list_.size() != 95) {
        // -1: wrong dgram count
        return -1;
    }
    // (2) check size of each dgram
    for (size_t i = 0; i < dgram_list_.size(); i++) {
        if (i == 0) {
            if (dgram_list_[i]->size() != 1380) {
                // -2: wrong first dgram size
                return -2;
            }
        } else {
            if (dgram_list_[i]->size() != 1384) {
                // -3: wrong other dgram size
                return -3;
            }
        }
    }
    // (3) check dgram order
    uint8_t target_mod_id = 0;
    uint16_t target_frm_sn = 0;
    uint8_t previous_seg_sn = 0;
    for (size_t i = 0; i < dgram_list_.size(); i++) {
        if (i == 0) {
            target_mod_id = gDC.decode_byte<uint8_t>(dgram_list_[i]->data(), 0, 0);
            target_frm_sn = gDC.decode_byte<uint16_t>(dgram_list_[i]->data(), 1, 2);
            previous_seg_sn = gDC.decode_byte<uint8_t>(dgram_list_[i]->data(), 3, 3);
        } else {
            uint8_t current_mod_id = gDC.decode_byte<uint8_t>(dgram_list_[i]->data(), 0, 0);
            uint16_t current_frm_sn = gDC.decode_byte<uint16_t>(dgram_list_[i]->data(), 1, 2);
            uint8_t current_seg_sn = gDC.decode_byte<uint8_t>(dgram_list_[i]->data(), 3, 3);
            if (current_mod_id == target_mod_id && current_frm_sn == target_frm_sn &&
                current_seg_sn == previous_seg_sn + 1) {
                previous_seg_sn = current_seg_sn;
            } else {
                // -4: found wrong order or mismatched dgrams
                return -4;
            }
        }
    }
    // (4) check crc
    crc_32_type crc_32;
    for (size_t i = 0; i < dgram_list_.size() - 1; i++) {
        crc_32.process_bytes(dgram_list_[i]->data() + 4, dgram_list_[i]->size() - 4);
    }
    size_t last_index = dgram_list_.size() - 1;
    size_t last_size = dgram_list_[last_index]->size();
    crc_32.process_bytes(dgram_list_[last_index]->data() + 4, dgram_list_[last_index]->size() - 8);
    uint32_t target_crc = gDC.decode_byte<uint32_t>(dgram_list_[last_index]->data(), last_size - 4, last_size - 1);
    if (target_crc != crc_32.checksum()) {
        // -5: wrong CRC
        return -5;
    }
    // return total size of payload on success
    return 131096;
}

void diffraflow::ImageFrameRaw::sort_dgrams() {
    //
}

shared_ptr<vector<char>>& diffraflow::ImageFrameRaw::get_dgram(size_t index) { return dgram_list_[index]; }

size_t diffraflow::ImageFrameRaw::get_dgram_count() { return dgram_list_.size(); }
