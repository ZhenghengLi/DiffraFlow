#ifndef __IngImgFtrBuffer_H__
#define __IngImgFtrBuffer_H__

#include <cstddef>

#include "ImageDataField.hh"
#include "ImageFeature.hh"

namespace diffraflow {
    class IngImgFtrBuffer {
    public:
        IngImgFtrBuffer();
        ~IngImgFtrBuffer();

    private:
        size_t head_;
        size_t tail_;
        size_t flag_;
    };
} // namespace diffraflow

#endif