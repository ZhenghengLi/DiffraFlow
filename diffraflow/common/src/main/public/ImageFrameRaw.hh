#ifndef __ImageFrameRaw_H__
#define __ImageFrameRaw_H__

#include <string>

namespace diffraflow {
    class ImageFrameRaw {
    public:
        ImageFrameRaw();
        ~ImageFrameRaw();

        bool set_data(const char* buffer, const size_t len);

        uint64_t get_key();
        char* data();
        size_t size();

    public:
        uint64_t bunch_id; // key
        int16_t module_id; // 0 -- 15

    private:
        char* data_buffer_;
        size_t data_size_;
    };
} // namespace diffraflow

#endif