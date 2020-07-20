#ifndef ImageData_H
#define ImageData_H

#include <iostream>
#include <vector>
#include <msgpack.hpp>

#include "ImageFrame.hh"

using std::vector;
using std::ostream;

namespace diffraflow {

    class ImageData {
    public:
        explicit ImageData(uint32_t numOfMod = 1);
        ~ImageData();

        bool put_imgfrm(size_t index, const ImageFrame& imgfrm);
        void print(ostream& out = std::cout) const;

        void set_defined();
        bool get_defined();

        void set_calib_level(int level);
        int get_calib_level();

        uint64_t get_key();
        void set_key(uint64_t key);

    public:
        uint64_t bunch_id;
        vector<bool> alignment_vec;
        vector<ImageFrame> image_frame_vec;
        bool late_arrived;

    private:
        bool is_defined_;
        int calib_level_;

    public:
        MSGPACK_DEFINE_MAP(bunch_id, alignment_vec, image_frame_vec, late_arrived, is_defined_, calib_level_);
    };

} // namespace diffraflow

#endif
