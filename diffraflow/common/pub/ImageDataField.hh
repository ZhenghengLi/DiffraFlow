#ifndef __ImageDataField_H__
#define __ImageDataField_H__

#define MOD_CNT 16
#define FRAME_H 512
#define FRAME_W 128
#define FRAME_L 65536
#define FRAME_S 131096

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