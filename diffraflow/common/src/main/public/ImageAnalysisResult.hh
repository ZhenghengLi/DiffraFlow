#ifndef __ImageAnalysisResult_H__
#define __ImageAnalysisResult_H__

#include "ImageDataFeature.hh"

#include <msgpack.hpp>

namespace diffraflow {
    struct ImageAnalysisResult {
        int int_result;
        float float_result;
        MSGPACK_DEFINE_MAP(int_result, float_result);
    };
} // namespace diffraflow

#endif