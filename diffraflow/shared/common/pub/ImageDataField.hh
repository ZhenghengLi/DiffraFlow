#ifndef __ImageDataField_H__
#define __ImageDataField_H__

#include "ImageDimension.hh"

namespace diffraflow {
    struct ImageDataField {
        float pixel_data[MOD_CNT][FRAME_H][FRAME_W];
        unsigned char gain_level[MOD_CNT][FRAME_H][FRAME_W];
        unsigned long bunch_id;
        bool alignment[MOD_CNT];
        short cell_id[MOD_CNT];
        unsigned short status[MOD_CNT];
        bool late_arrived;
        char calib_level;
    };
}; // namespace diffraflow

#endif