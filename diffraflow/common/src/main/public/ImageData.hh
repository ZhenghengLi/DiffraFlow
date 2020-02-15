#ifndef ImageData_H
#define ImageData_H

#include <vector>
#include <msgpack.hpp>

#include "ImageFrame.hh"

using std::vector;

namespace diffraflow {

    class ImageData {
    public:
        explicit ImageData(uint32_t numOfDet = 1);
        ~ImageData();

        bool put_imgfrm(size_t index, const ImageFrame& imgfrm);
        void print();

    public:
        int64_t            event_key;   // equal to image_key
        double             event_time;  // equal to image_time
        vector<ImageFrame> imgfrm_vec;  // image data from each sub-detector

    public:
        MSGPACK_DEFINE (
            event_key,
            event_time,
            imgfrm_vec
        );
    };
}

#endif
