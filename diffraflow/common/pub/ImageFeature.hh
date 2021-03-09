#ifndef __ImageFeature_H__
#define __ImageFeature_H__

#ifndef __CUDACC__
#include <msgpack.hpp>
#endif

namespace diffraflow {
    struct ImageFeature {
        double global_rms;
        int peak_counts;
#ifndef __CUDACC__
        MSGPACK_DEFINE_MAP(global_rms, peak_counts);
#endif
    };
} // namespace diffraflow

#endif