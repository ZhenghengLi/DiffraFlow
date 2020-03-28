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
        uint64_t           event_time;  // equal to image_time
        vector<bool>       status_vec;  // alignment status for each sub-detector
        vector<ImageFrame> imgfrm_vec;  // image data from each sub-detector
        uint64_t           wait_threshold;
        bool               late_arrived;

    public:
        MSGPACK_DEFINE_MAP (
            event_time,
            status_vec,
            imgfrm_vec,
            wait_threshold,
            late_arrived
        );
    };
}

#endif
