#ifndef ImageData_H
#define ImageData_H

#include <iostream>
#include <vector>
#include <memory>
#include <msgpack.hpp>

#include "ImageFrame.hh"

using std::vector;
using std::ostream;
using std::shared_ptr;
using std::make_shared;

namespace diffraflow {

    class ImageData {
    public:
        explicit ImageData(uint32_t numOfMod = 0);
        ~ImageData();

        bool put_imgfrm(size_t index, const shared_ptr<ImageFrame>& imgfrm);
        void print(ostream& out = std::cout) const;

        uint64_t get_key();

        bool decode(const char* buffer, const size_t len);

    public:
        uint64_t bunch_id;
        vector<bool> alignment_vec;
        vector<shared_ptr<ImageFrame>> image_frame_vec;
        bool late_arrived;
        int calib_level;

    public:
        MSGPACK_DEFINE_MAP(bunch_id, alignment_vec, image_frame_vec, late_arrived, calib_level);
    };

} // namespace diffraflow

#endif
