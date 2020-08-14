#include "ImageDataRaw.hh"
#include "ImageFrameRaw.hh"

diffraflow::ImageDataRaw::ImageDataRaw(uint32_t numOfMod) {
    bunch_id = 0;
    alignment_vec.resize(numOfMod, false);
    image_frame_vec.resize(numOfMod);
    late_arrived = false;
}

diffraflow::ImageDataRaw::~ImageDataRaw() {}

uint64_t diffraflow::ImageDataRaw::get_key() { return bunch_id; }

void diffraflow::ImageDataRaw::set_key(uint64_t key) { bunch_id = key; }

bool diffraflow::ImageDataRaw::put_imgfrm(size_t index, const shared_ptr<ImageFrameRaw>& imgfrm) {
    if (index >= image_frame_vec.size()) return false;
    alignment_vec[index] = true;
    image_frame_vec[index] = imgfrm;
    return true;
}
