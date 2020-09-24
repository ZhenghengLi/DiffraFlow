#ifndef __ImageVisObject_H__
#define __ImageVisObject_H__

#include "ImageDataSmall.hh"
#include "ImageFeature.hh"
#include "ImageAnalysisResult.hh"

#include <memory>
#include <msgpack.hpp>

using std::shared_ptr;

namespace diffraflow {
    class ImageVisObject {
    public:
        shared_ptr<ImageDataSmall> image_data;
        shared_ptr<ImageFeature> image_feature;
        shared_ptr<ImageAnalysisResult> analysis_result;

    public:
        MSGPACK_DEFINE_MAP(image_data, image_feature, analysis_result);
    };
} // namespace diffraflow

#endif