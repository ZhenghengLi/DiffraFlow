#include "ImageDataSmall.hh"
#include "ImageData.hh"
#include <numeric>

using std::numeric_limits;
using std::make_shared;

diffraflow::ImageDataSmall::ImageDataSmall() {}

diffraflow::ImageDataSmall::ImageDataSmall(const ImageData& image_data, float energy_down_cut, float energy_up_cut) {
    copy_from_(image_data, energy_down_cut, energy_up_cut);
}

diffraflow::ImageDataSmall::~ImageDataSmall() {}

diffraflow::ImageDataSmall& diffraflow::ImageDataSmall::operator=(const ImageData& image_data) {
    copy_from_(image_data, numeric_limits<float>::min(), numeric_limits<float>::max());
}

void diffraflow::ImageDataSmall::copy_from_(const ImageData& image_data, float energy_down_cut, float energy_up_cut) {
    bunch_id = image_data.bunch_id;
    alignment_vec = image_data.alignment_vec;
    max_energy = numeric_limits<float>::min();
    min_energy = numeric_limits<float>::max();
    for (size_t i = 0; i < image_data.image_frame_vec.size(); i++) {
        if (!image_data.image_frame_vec[i]) continue;
        ImageFrame& image_frame = *image_data.image_frame_vec[i];
        for (size_t j = 0; j < image_frame.pixel_data.size(); j++) {
            if (image_frame.pixel_data[j] > max_energy) max_energy = image_frame.pixel_data[j];
            if (image_frame.pixel_data[j] < min_energy) min_energy = image_frame.pixel_data[j];
        }
    }
    if (min_energy < energy_down_cut) min_energy = energy_down_cut;
    if (max_energy > energy_up_cut) max_energy = energy_up_cut;
    image_frame_vec.resize(image_data.image_frame_vec.size());
    float gap_energy = max_energy - min_energy;
    for (size_t i = 0; i < image_data.image_frame_vec.size(); i++) {
        if (image_data.image_frame_vec[i]) {
            ImageFrame& image_frame = *image_data.image_frame_vec[i];
            image_frame_vec[i] = make_shared<vector<uint8_t>>();
            image_frame_vec[i]->resize(image_frame.pixel_data.size());
            for (size_t j = 0; j < image_frame.pixel_data.size(); j++) {
                float energy = image_frame.pixel_data[i] - min_energy;
                if (energy < 0) energy = 0;
                if (energy > gap_energy) energy = gap_energy;
                image_frame_vec[i]->at(j) = (uint8_t)(256 * energy / gap_energy);
            }
        } else {
            image_frame_vec[i] = nullptr;
        }
    }
    late_arrived = image_data.late_arrived;
    calib_level = image_data.calib_level;
}
