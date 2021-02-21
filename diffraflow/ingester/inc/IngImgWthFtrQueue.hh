#ifndef __IngImgWthFtrQueue_H__
#define __IngImgWthFtrQueue_H__

#include "BlockingQueue.hh"
#include "ImageWithFeature.hh"

namespace diffraflow {
    typedef BlockingQueue<shared_ptr<ImageWithFeature>> IngImgWthFtrQueue;
} // namespace diffraflow

#endif