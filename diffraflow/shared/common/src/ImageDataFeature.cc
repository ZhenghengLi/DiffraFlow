#include "ImageDataFeature.hh"
#include "ImageWithFeature.hh"
#include "ImageDataType.hh"
#include "ImageDataField.hh"

diffraflow::ImageDataFeature::ImageDataFeature() {}

diffraflow::ImageDataFeature::ImageDataFeature(ImageDataField* image_data_ptr, ImageFeature* image_feature_ptr) {
    if (image_data_ptr) {
        this->image_data = make_shared<ImageData>();
        ImageDataType::convert(*image_data_ptr, *this->image_data);
    } else {
        this->image_data = nullptr;
    }
    if (image_feature_ptr) {
        this->image_feature = make_shared<ImageFeature>();
        *this->image_feature = *image_feature_ptr;
    } else {
        this->image_feature = nullptr;
    }
}
