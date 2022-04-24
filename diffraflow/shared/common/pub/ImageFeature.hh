#ifndef __ImageFeature_H__
#define __ImageFeature_H__

#ifndef __CUDACC__
#include <msgpack.hpp>
#endif

namespace diffraflow {
    struct ImageFeature {

        // energy mean of all pixels except those at vertical ASIC edge
        float global_mean;

        // energy rms of all pixels except those at vertical ASIC edge
        float global_rms;

        // total number of pixels that belong to peaks (i.e. outliers)
        int peak_pixels;

#ifndef __CUDACC__
        MSGPACK_DEFINE_MAP(global_rms, global_mean, peak_pixels)
#endif

        void clear() {
            global_mean = 0;
            global_rms = 0;
            peak_pixels = 0;
        }
    };
} // namespace diffraflow

#endif