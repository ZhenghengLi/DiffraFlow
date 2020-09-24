#ifndef __ImageDataSmall_H__
#define __ImageDataSmall_H__

#include <vector>
#include <memory>
#include <msgpack.hpp>

using std::vector;
using std::shared_ptr;

namespace diffraflow {

    class ImageData;

    class ImageDataSmall {
    public:
        ImageDataSmall();
        explicit ImageDataSmall(
            const ImageData& image_data, float energy_down_cut = -1000000, float energy_up_cut = 1000000);
        ~ImageDataSmall();

        ImageDataSmall& operator=(const ImageData& image_data);

    public:
        uint64_t bunch_id;
        vector<bool> alignment_vec;
        vector<shared_ptr<vector<uint8_t>>> image_frame_vec;
        bool late_arrived;
        int calib_level;
        float max_energy;
        float min_energy;

    public:
        MSGPACK_DEFINE_MAP(bunch_id, alignment_vec, image_frame_vec, late_arrived, calib_level, max_energy, min_energy);

    private:
        void copy_from_(const ImageData& image_data, float energy_down_cut, float energy_up_cut);
    };
} // namespace diffraflow

#endif