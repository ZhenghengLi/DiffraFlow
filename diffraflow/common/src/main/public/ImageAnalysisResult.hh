#ifndef __ImageAnalysisResult_H__
#define __ImageAnalysisResult_H__

#include "ImageWithFeature.hh"

#include <msgpack.hpp>

namespace diffraflow {
    struct ImageAnalysisResult {

        ImageWithFeature image_with_feature;
        // analysis results
        int int_result;
        float float_result;

        MSGPACK_DEFINE_MAP(image_with_feature, int_result, float_result);
    };
} // namespace diffraflow

#endif