#ifndef __ImageDataHDF5_H__
#define __ImageDataHDF5_H__

#include "H5Cpp.h"
#include "ImageData.hh"

#define MOD_CNT 16
#define IMAGE_H 512
#define IMAGE_W 128

namespace diffraflow {

    class ImageDataHDF5 : public H5::CompType {
    public:
        ImageDataHDF5();
        ~ImageDataHDF5();

    public:
        struct Field {
            uint64_t bunch_id;
            bool alignment[MOD_CNT];
            int16_t cell_id[MOD_CNT];
            uint16_t status[MOD_CNT];
            float pixel_data[MOD_CNT][IMAGE_H][IMAGE_W];
            uint8_t gain_level[MOD_CNT][IMAGE_H][IMAGE_W];
            bool late_arrived;
        };

    public:
        static void convert_image(const ImageData& imgdat_obj, Field& imgdat_st);
    };
} // namespace diffraflow

#endif