#ifndef __ImageFramePtr_H__
#define __ImageFramePtr_H__

#include "ImageFrame.hh"
#include <memory>

using std::shared_ptr;
using std::make_shared;

namespace diffraflow {
    class ImageFramePtr: public shared_ptr<ImageFrame> {
    public:
        ImageFramePtr(ImageFrame* image_frame): shared_ptr<ImageFrame>(image_frame) { }

        ImageFramePtr() {
            ImageFramePtr(new ImageFrame());
        }

        bool operator<(const ImageFramePtr& right) const {
            return (*this)->image_time < right->image_time;
        }

        bool operator<=(const ImageFramePtr& right) const {
            return (*this)->image_time <= right->image_time;
        }

        bool operator>(const ImageFramePtr& right) const {
            return (*this)->image_time > right->image_time;
        }

        bool operator>=(const ImageFramePtr& right) const {
            return (*this)->image_time >= right->image_time;
        }

        bool operator==(const ImageFramePtr& right) const {
            return (*this)->image_time == right->image_time;
        }

        int64_t operator-(const ImageFramePtr& right) const {
            return (*this)->image_time - right->image_time;
        }

    };
}

#endif