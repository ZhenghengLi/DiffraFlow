#ifndef __ImageDataFeature_H__
#define __ImageDataFeature_H__

#include "ImageData.hh"
#include "ImageDataField.hh"
#include "ImageFeature.hh"

#include <memory>
#include <msgpack.hpp>

using std::shared_ptr;

namespace diffraflow {

    class ImageDataFeature {
    public:
        ImageDataFeature();
        ImageDataFeature(ImageDataField* image_data_ptr, ImageFeature* image_feature_ptr);

    public:
        // image data
        shared_ptr<ImageData> image_data;
        // image feature
        shared_ptr<ImageFeature> image_feature;

    public:
        MSGPACK_DEFINE_MAP(image_data, image_feature);
    };
} // namespace diffraflow

#endif