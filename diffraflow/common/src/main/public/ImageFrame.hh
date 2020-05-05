#ifndef ImageFrame_H
#define ImageFrame_H

#include <vector>
#include <msgpack.hpp>
#include <log4cxx/logger.h>

using std::vector;

namespace diffraflow {
    class ImageFrame {
    public:
        ImageFrame();
        ~ImageFrame();

        bool decode(const char* buffer, const size_t size);
        void print() const;

    public:
        uint64_t image_time; // unit: nanosecond
        int32_t detector_id;
        uint32_t image_width;
        uint32_t image_height;
        vector<float> image_frame; // size = width * height;

        vector<char> image_rawdata;

    public:
        MSGPACK_DEFINE_MAP(image_time, detector_id, image_width, image_height, image_frame, image_rawdata);

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
