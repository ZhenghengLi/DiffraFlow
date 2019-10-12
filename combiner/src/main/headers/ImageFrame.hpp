#ifndef ImageFrame_H
#define ImageFrame_H

#include <iostream>

namespace shine {
    class ImageFrame {
    private:
        char*  img_rawdata_;
        size_t img_rawsize_;

    public:
        int64_t img_key;
        int     det_id;
        double  img_time;     // unit: second
        int     img_width;
        int     img_height;
        float*  img_frame;    // size = width * height;

    public:
        ImageFrame();
        // copy constructor
        ImageFrame(const ImageFrame& img_frm);
        ~ImageFrame();

        bool decode(const char* img_rd, const size_t img_sz);
        void print();

    };
}

#endif