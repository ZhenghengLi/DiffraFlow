#ifndef __ImageDataRaw_H__
#define __ImageDataRaw_H__

#include <iostream>
#include <vector>
#include <memory>

using std::vector;
using std::shared_ptr;
using std::ostream;

namespace diffraflow {

    class ImageFrameRaw;

    class ImageDataRaw {
    public:
        explicit ImageDataRaw(uint32_t numOfMod = 1);
        ~ImageDataRaw();

        bool put_imgfrm(size_t index, const shared_ptr<ImageFrameRaw>& imgfrm);
        void print(ostream& out = std::cout) const;

        uint64_t get_key();
        void set_key(uint64_t key);

        size_t serialize_meta(char* buffer, size_t len) const;

    public:
        uint64_t bunch_id;
        vector<bool> alignment_vec;
        bool late_arrived;
        vector<shared_ptr<ImageFrameRaw>> image_frame_vec;
    };
} // namespace diffraflow

#endif