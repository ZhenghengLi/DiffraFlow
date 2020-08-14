#ifndef __ImageFrameRaw_H__
#define __ImageFrameRaw_H__

#include <vector>
#include <memory>

using std::vector;
using std::shared_ptr;
using std::make_shared;

namespace diffraflow {
    class ImageFrameRaw {
    public:
        ImageFrameRaw();
        ~ImageFrameRaw();

        bool set_data(const shared_ptr<vector<char>>& buffer);

        uint64_t get_key();
        char* data();
        size_t size();

    public:
        uint64_t bunch_id; // key
        int16_t module_id; // 0 -- 15

    private:
        shared_ptr<vector<char>> data_buffer_;
    };
} // namespace diffraflow

#endif