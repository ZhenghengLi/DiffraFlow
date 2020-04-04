#ifndef __ImageFeature_H__
#define __ImageFeature_H__

#include <msgpack.hpp>

namespace diffraflow {
    class ImageFeature {
    public:
        ImageFeature();
        ~ImageFeature();

    public:
        // feature list
        double global_rms;
        int peak_counts;

    public:
        MSGPACK_DEFINE_MAP(
            global_rms,
            peak_counts
        );

    };
}

#endif