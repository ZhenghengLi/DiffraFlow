#ifndef ImageCache_H
#define ImageCache_H

#include "BlockingQueue.hh"

namespace shine {

    class ImageFrame;
    class ImageData;

    class ImageCache {
    private:
        BlockingQueue<ImageData> image_data_queue_;

    public:
        ImageCache();
        ~ImageCache();

        void put_frame(ImageFrame& image_frame);
        bool take_one_image(ImageData& image_data, int timeout_ms);
        void img_queue_stop();
        bool img_queue_stopped();

    };
}

#endif