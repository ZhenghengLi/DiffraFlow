#ifndef __ImageFeature_H__
#define __ImageFeature_H__

#include <msgpack.hpp>

namespace diffraflow {
    struct ImageFeature {
        float global_rms;
        int peak_counts;
        MSGPACK_DEFINE_MAP(global_rms, peak_counts);
        void clear() {
            global_rms = 0;
            peak_counts = 0;
        }
    };
} // namespace diffraflow

#endif