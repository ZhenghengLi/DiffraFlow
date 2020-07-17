#ifndef ImageFrame_H
#define ImageFrame_H

#include <vector>
#include <iostream>
#include <msgpack.hpp>
#include <log4cxx/logger.h>

using std::vector;
using std::ostream;

namespace diffraflow {
    class ImageFrame {
    public:
        ImageFrame();
        ~ImageFrame();

        bool decode(const char* buffer, const size_t size);
        void print(ostream& out = std::cout) const;

    public:
        uint64_t bunch_id;          // key
        int16_t module_id;          // 0 -- 15
        int16_t cell_id;            // 0 -- 351
        uint16_t status;            // 0
        vector<float> pixel_data;   // size = 512 * 128
        vector<uint8_t> gain_level; // size = 512 * 128

    public:
        MSGPACK_DEFINE_MAP(bunch_id, module_id, cell_id, status, pixel_data, gain_level);

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
