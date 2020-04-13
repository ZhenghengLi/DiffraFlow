#ifndef __ImageDataHDF5_H__
#define __ImageDataHDF5_H__

#include <H5Cpp.h>
#include "ImageData.hh"

#define DET_CNT 4
#define IMAGE_H 6
#define IMAGE_W 8

namespace diffraflow {

    class ImageDataHDF5: public H5::CompType {
    public:
        ImageDataHDF5();
        ~ImageDataHDF5();

    public:
        struct Field {
            uint64_t    event_time;
            bool        alignment[DET_CNT];
            float       image_frame[DET_CNT][IMAGE_H][IMAGE_W];
            uint64_t    wait_threshold;
            bool        late_arrived;
        };

    public:
        static void convert_image(const ImageData& imgdat_obj, Field& imgdat_st);

    };
}

#endif