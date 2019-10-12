#ifndef ImageCache_H
#define ImageCache_H

namespace shine {

    class ImageFrame;
    class ImageData;

    class ImageCache {
    private:

    public:
        ImageCache();
        ~ImageCache();

        void put_frame(const ImageFrame& image_frame);

        ImageData take_one_image();



    };
}

#endif