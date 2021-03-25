#ifndef __ImageDataType_H__
#define __ImageDataType_H__

#include "H5Cpp.h"
#include <iostream>

#include "ImageDataField.hh"

using std::ostream;

namespace diffraflow {

    class ImageData;

    class ImageDataType : public H5::CompType {
    public:
        ImageDataType();
        ~ImageDataType();

    public:
        static bool decode(ImageDataField* image_data, const char* buffer, const size_t len);
        static void print(const ImageDataField& image_data, ostream& out = std::cout);
        static void convert(const ImageDataField& image_data_arr, ImageData& image_data_obj);
    };
} // namespace diffraflow

#endif