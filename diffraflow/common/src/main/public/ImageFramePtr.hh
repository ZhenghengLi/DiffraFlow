#ifndef __ImageFramePtr_H__
#define __ImageFramePtr_H__

#include "ImageFrame.hh"
#include <memory>

using std::shared_ptr;
using std::make_shared;

namespace diffraflow {
    class ImageFramePtr : public shared_ptr<ImageFrame> {
    public:
        explicit ImageFramePtr(ImageFrame* image_frame) : shared_ptr<ImageFrame>(image_frame) {}

        ImageFramePtr() : shared_ptr<ImageFrame>() {}

        bool operator<(const ImageFramePtr& right) const {
            if ((*this) && right) {
                return (*this)->bunch_id < right->bunch_id;
            } else {
                return false;
            }
        }

        bool operator<=(const ImageFramePtr& right) const {
            if ((*this) && right) {
                return (*this)->bunch_id <= right->bunch_id;
            } else {
                return false;
            }
        }

        bool operator>(const ImageFramePtr& right) const {
            if ((*this) && right) {
                return (*this)->bunch_id > right->bunch_id;
            } else {
                return false;
            }
        }

        bool operator>=(const ImageFramePtr& right) const {
            if ((*this) && right) {
                return (*this)->bunch_id >= right->bunch_id;
            } else {
                return false;
            }
        }

        bool operator==(const ImageFramePtr& right) const {
            if ((*this) && right) {
                return (*this)->bunch_id == right->bunch_id;
            } else {
                return false;
            }
        }

        int64_t operator-(const ImageFramePtr& right) const {
            if ((*this) && right) {
                return (*this)->bunch_id - right->bunch_id;
            } else {
                return 0;
            }
        }
    };
} // namespace diffraflow

#endif