#include "ImageDataFeature.hh"
#include "ImageWithFeature.hh"
#include "ImageDataType.hh"

diffraflow::ImageDataFeature::ImageDataFeature() {}

diffraflow::ImageDataFeature::ImageDataFeature(const ImageWithFeature& image_with_feature) {
    copy_from_(image_with_feature);
}

const diffraflow::ImageDataFeature& diffraflow::ImageDataFeature::operator=(
    const ImageWithFeature& image_with_feature) {
    copy_from_(image_with_feature);
    return *this;
}

void diffraflow::ImageDataFeature::copy_from_(const ImageWithFeature& image_with_feature) {
    if (image_with_feature.image_data) {
        this->image_data = make_shared<ImageData>();
        ImageDataType::convert(*image_with_feature.image_data, *this->image_data);
    } else {
        this->image_data = nullptr;
    }
    this->image_feature = image_with_feature.image_feature;
}