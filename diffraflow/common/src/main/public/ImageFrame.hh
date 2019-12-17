#ifndef ImageFrame_H
#define ImageFrame_H

#include <iostream>
#include "Decoder.hh"
#include "PrimitiveSerializer.hh"
#include "ObjectSerializer.hh"

namespace diffraflow {
    class ImageFrame: public ObjectSerializer {
    private:
        static const int obj_type_ = 1231;

    private:
        char*    img_rawdata_;
        uint32_t img_rawsize_;

    public:
        int64_t img_key;
        double  img_time;     // unit: second
        uint32_t det_id;
        uint32_t img_width;
        uint32_t img_height;
        float*  img_frame;    // size = width * height;

    private:
        void copyObj_(const ImageFrame& img_frm);

    public:
        ImageFrame();
        ImageFrame(const char* buffer, const size_t size);
        // copy constructor
        ImageFrame(const ImageFrame& img_frm);
        ~ImageFrame();

        ImageFrame& operator=(const ImageFrame& img_frm);

        bool decode(const char* buffer, const size_t size);
        void print();

        size_t serialize(char* const data, size_t len);
        size_t deserialize(const char* const data, size_t len);
        size_t object_size();
        int object_type();
        void clear_data();

    };
}

#endif
