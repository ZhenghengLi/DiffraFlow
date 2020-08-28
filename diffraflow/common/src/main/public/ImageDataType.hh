#ifndef __ImageDataType_H__
#define __ImageDataType_H__

#include "H5Cpp.h"
#include <iostream>

#define MOD_CNT 16
#define FRAME_H 512
#define FRAME_W 128
#define FRAME_L 65536
#define FRAME_S 131096

using std::ostream;

namespace diffraflow {

    class ImageDataType : public H5::CompType {
    public:
        ImageDataType();
        ~ImageDataType();

    public:
        struct Field {
            uint64_t bunch_id;
            bool alignment[MOD_CNT];
            int16_t cell_id[MOD_CNT];
            uint16_t status[MOD_CNT];
            float pixel_data[MOD_CNT][FRAME_H][FRAME_W];
            uint8_t gain_level[MOD_CNT][FRAME_H][FRAME_W];
            bool late_arrived;
            int8_t calib_level;
        };

    public:
        static bool decode(Field& image_data, const char* buffer, const size_t len);
        static void print(const Field& image_data, ostream& out = std::cout);
    };
} // namespace diffraflow

#endif