#ifndef __ImageDataFeature_H__
#define __ImageDataFeature_H__

#include "ImageData.hh"
#include "ImageFeature.hh"

#include <msgpack.hpp>

namespace diffraflow {

    class ImageWithFeature;

    class ImageDataFeature {
    public:
        ImageDataFeature();
        explicit ImageDataFeature(const ImageWithFeature& image_with_feature);

        const ImageDataFeature& operator=(const ImageWithFeature& image_with_feature);

    public:
        // image data
        ImageData image_data;
        // image feature
        ImageFeature image_feature;

    public:
        MSGPACK_DEFINE_MAP(image_data, image_feature);
    };
} // namespace diffraflow

#endif