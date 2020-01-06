#ifndef CmbImgCache_H
#define CmbImgCache_H

#include <queue>
#include "BlockingQueue.hh"

using std::queue;

namespace diffraflow {

    class ImageFrame;
    class ImageData;

    class CmbImgCache {
    public:
        explicit CmbImgCache(size_t num_of_dets);
        ~CmbImgCache();

        void put_frame(const ImageFrame& image_frame);
        bool do_alignment();
        bool take_one_image(ImageData& image_data);
        void img_queue_stop();
        bool img_queue_stopped();

    private:
        size_t              imgfrm_queues_len_;
        queue<ImageFrame>*  imgfrm_queues_arr_;

        BlockingQueue<ImageData> imgdat_queue_;

    };
}

#endif
