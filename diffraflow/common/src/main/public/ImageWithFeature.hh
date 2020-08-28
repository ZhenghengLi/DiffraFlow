#ifndef __ImageWithFeature_H__
#define __ImageWithFeature_H__

#include "ImageDataType.hh"
#include "ImageFeature.hh"

namespace diffraflow {
    struct ImageWithFeature {
        // image data
        ImageDataType::Field image_data;
        // image feature
        ImageFeature image_feature;
    };
} // namespace diffraflow

#endif