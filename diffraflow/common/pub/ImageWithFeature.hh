#ifndef __ImageWithFeature_H__
#define __ImageWithFeature_H__

#include "ImageDataType.hh"
#include "ImageFeature.hh"

#include <vector>
#include <memory>

using std::vector;
using std::shared_ptr;

namespace diffraflow {
    struct ImageWithFeature {
        // raw image data
        shared_ptr<vector<char>> image_data_raw;
        // image data
        shared_ptr<ImageDataField> image_data;
        // image feature
        shared_ptr<ImageFeature> image_feature;

        // for GPU
        // image data
        ImageDataField* image_data_devptr = 0;
        // image feature
        ImageFeature* image_feature_devptr = 0;
    };
} // namespace diffraflow

#endif