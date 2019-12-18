#ifndef CmbImgCache_H
#define CmbImgCache_H

#include "BlockingQueue.hh"

namespace diffraflow {

    class ImageFrame;
    class ImageData;

    class CmbImgCache {
    private:
        BlockingQueue<ImageData> image_data_queue_;

    public:
        CmbImgCache();
        ~CmbImgCache();

        void put_frame(ImageFrame& image_frame);
        bool take_one_image(ImageData& image_data, int timeout_ms);
        void img_queue_stop();
        bool img_queue_stopped();

    };
}

#endif
