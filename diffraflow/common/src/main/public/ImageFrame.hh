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

        bool operator<(const ImageFrame& right) const;
        bool operator<=(const ImageFrame& right) const;
        bool operator>(const ImageFrame& right) const;
        bool operator>=(const ImageFrame& right) const;
        bool operator==(const ImageFrame& right) const;
        double operator-(const ImageFrame& right) const;

    public:
        int64_t       img_key;
        double        img_time;     // unit: second
        int32_t       det_id;
        int32_t       img_width;
        int32_t       img_height;
        vector<float> img_frame;    // size = width * height;

    private:
        vector<char>  img_rawdata_;

    public:
        MSGPACK_DEFINE_MAP (
            img_rawdata_,
            img_key,
            img_time,
            det_id,
            img_width,
            img_height,
            img_frame
        );

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif
