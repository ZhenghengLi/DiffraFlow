#ifndef __ImageFileHDF5_H__
#define __ImageFileHDF5_H__

#include <H5Cpp.h>

#define DET_CNT 4
#define IMAGE_H 6
#define IMAGE_W 8

namespace diffraflow {
    class ImageFileHDF5 {
    public:
        ImageFileHDF5(size_t chunk_size, size_t batch_size, int compress_level = -1);
        ~ImageFileHDF5();

        bool open_write(const char* filename);
        bool open_read(const char* filename);

    public:
        struct ImageDataS {
            uint64_t    event_time;
            bool        alignment[DET_CNT];
            float       image_frame[DET_CNT][IMAGE_H][IMAGE_W];
            uint64_t    wait_threshold;
            bool        late_arrived;
        };

    private:
        size_t chunk_size_;
        size_t batch_size_;
        int compress_level_;

    };
}

#endif