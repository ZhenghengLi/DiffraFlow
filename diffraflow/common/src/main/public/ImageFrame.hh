#ifndef ImageFrame_H
#define ImageFrame_H

#include <iostream>
#include "Decoder.hh"

namespace shine {
    class ImageFrame: private Decoder {
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
        ImageFrame(const char* buffer, const size_t size);
        // copy constructor
        ImageFrame(const ImageFrame& img_frm);
        ~ImageFrame();

        bool decode(const char* buffer, const size_t size);
        void print();

    };
}

#endif