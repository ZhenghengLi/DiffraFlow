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
        ImageDataType::Field image_data;
        // image feature
        ImageFeature image_feature;
    };
} // namespace diffraflow

#endif