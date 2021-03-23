#ifndef __ImageDataFeature_H__
#define __ImageDataFeature_H__

#include "ImageData.hh"
#include "ImageFeature.hh"

#include <memory>
#include <msgpack.hpp>

using std::shared_ptr;

namespace diffraflow {

    class ImageWithFeature;
    class ImageDataField;

    class ImageDataFeature {
    public:
        ImageDataFeature();
        ImageDataFeature(ImageDataField* image_data_ptr, ImageFeature* image_feature_ptr);
        explicit ImageDataFeature(const ImageWithFeature& image_with_feature);

        const ImageDataFeature& operator=(const ImageWithFeature& image_with_feature);

    public:
        // image data
        shared_ptr<ImageData> image_data;
        // image feature
        shared_ptr<ImageFeature> image_feature;

    public:
        MSGPACK_DEFINE_MAP(image_data, image_feature);

    private:
        void copy_from_(const ImageWithFeature& image_with_feature);
    };
} // namespace diffraflow

#endif