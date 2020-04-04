#ifndef __ImageFeature_H__
#define __ImageFeature_H__

#include <msgpack.hpp>

namespace diffraflow {
    class ImageFeature {
    public:
        ImageFeature();
        ~ImageFeature();

        void set_defined();
        bool get_defined();

    public:
        // feature list
        double global_rms;
        int peak_counts;

    private:
        bool is_defined_;

    public:
        MSGPACK_DEFINE_MAP(
            global_rms,
            peak_counts,
            is_defined_
        );

    };
}

#endif