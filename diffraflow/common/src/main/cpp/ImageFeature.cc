#include "ImageFeature.hh"

diffraflow::ImageFeature::ImageFeature() {
    global_rms = 0;
    peak_counts = 0;
    is_defined_ = false;
}

diffraflow::ImageFeature::~ImageFeature() {}

void diffraflow::ImageFeature::set_defined() { is_defined_ = true; }

bool diffraflow::ImageFeature::get_defined() { return is_defined_; }
