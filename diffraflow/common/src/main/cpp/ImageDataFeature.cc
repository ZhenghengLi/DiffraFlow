#include "ImageDataFeature.hh"
#include "ImageWithFeature.hh"
#include "ImageDataType.hh"

diffraflow::ImageDataFeature::ImageDataFeature() {}

diffraflow::ImageDataFeature::ImageDataFeature(const ImageWithFeature& image_with_feature) {
    ImageDataType::convert(image_with_feature.image_data, this->image_data);
    this->image_feature = image_with_feature.image_feature;
}

const diffraflow::ImageDataFeature& diffraflow::ImageDataFeature::operator=(
    const ImageWithFeature& image_with_feature) {
    ImageDataType::convert(image_with_feature.image_data, this->image_data);
    this->image_feature = image_with_feature.image_feature;
    return *this;
}
