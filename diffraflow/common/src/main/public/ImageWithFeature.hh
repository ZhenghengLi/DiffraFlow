#ifndef __ImageWithFeature_H__
#define __ImageWithFeature_H__

#include "ImageData.hh"
#include "ImageFeature.hh"

#include <msgpack.hpp>

namespace diffraflow {
    struct ImageWithFeature {

        // image data
        ImageData image_data_raw;
        ImageData image_data_calib;
        // image feature
        ImageFeature image_feature;

        MSGPACK_DEFINE_MAP(
            image_data_raw,
            image_data_calib,
            image_feature
        );

    };
}

#endif