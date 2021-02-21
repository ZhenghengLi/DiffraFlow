#include "ImageDataRaw.hh"
#include "ImageFrameRaw.hh"
#include "PrimitiveSerializer.hh"

using std::endl;
using std::cout;
using std::string;

diffraflow::ImageDataRaw::ImageDataRaw(uint32_t numOfMod) {
    bunch_id = 0;
    alignment_vec.resize(numOfMod, false);
    image_frame_vec.resize(numOfMod);
    late_arrived = false;
}

diffraflow::ImageDataRaw::~ImageDataRaw() {}

size_t diffraflow::ImageDataRaw::serialize_meta(char* buffer, size_t len) const {
    if (len < 11) return 0;
    if (alignment_vec.size() != 16) return 0;
    size_t current_pos = 0;
    // bunch_id
    gPS.serializeValue<uint64_t>(bunch_id, buffer + current_pos, 8);
    current_pos += 8;
    // alignment_vec
    buffer[current_pos] = 0;
    buffer[current_pos + 1] = 0;
    for (size_t i = 0; i < 16; i++) {
        size_t pos = current_pos + i / 8;
        size_t bit = i % 8;
        if (alignment_vec[i]) {
            buffer[pos] |= (1 << (7 - bit));
        }
    }
    current_pos += 2;
    // late_arrived
    gPS.serializeValue<uint8_t>((uint8_t)late_arrived, buffer + current_pos, 1);
    current_pos += 1;
    return current_pos;
}

uint64_t diffraflow::ImageDataRaw::get_key() { return bunch_id; }

void diffraflow::ImageDataRaw::set_key(uint64_t key) { bunch_id = key; }

bool diffraflow::ImageDataRaw::put_imgfrm(size_t index, const shared_ptr<ImageFrameRaw>& imgfrm) {
    if (index >= image_frame_vec.size()) return false;
    alignment_vec[index] = true;
    image_frame_vec[index] = imgfrm;
    return true;
}

void diffraflow::ImageDataRaw::print(ostream& out) const {
    out << "bunch_id: " << bunch_id << endl;
    out << "late_arrived: " << late_arrived << endl;
    out << "alignment_vec: [";
    for (size_t i = 0; i < alignment_vec.size(); i++) {
        if (i > 0) out << ", ";
        out << alignment_vec[i];
    }
    out << "]" << endl;
}
