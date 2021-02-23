#ifndef __ImageFeature_H__
#define __ImageFeature_H__

#include <msgpack.hpp>

namespace diffraflow {
    struct ImageFeature {
        double global_rms;
        int peak_counts;
        MSGPACK_DEFINE_MAP(global_rms, peak_counts);
    };
} // namespace diffraflow

#endif