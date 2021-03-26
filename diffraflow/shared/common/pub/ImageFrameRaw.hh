#ifndef __ImageFrameRaw_H__
#define __ImageFrameRaw_H__

#include <vector>
#include <memory>

#include "ByteBuffer.hh"

using std::vector;
using std::shared_ptr;
using std::make_shared;

namespace diffraflow {
    class ImageFrameRaw {
    public:
        ImageFrameRaw();
        ~ImageFrameRaw();

        bool set_data(const shared_ptr<ByteBuffer>& buffer);
        char* data();
        size_t size();

        bool add_dgram(shared_ptr<ByteBuffer>& dgram);
        void sort_dgrams();
        int check_dgrams_integrity();
        shared_ptr<ByteBuffer>& get_dgram(size_t index);
        size_t get_dgram_count();

        uint64_t get_key();

    public:
        uint64_t bunch_id; // key
        int16_t module_id; // 0 -- 15

        uint8_t dgram_mod_id;
        uint16_t dgram_frm_sn;

    private:
        shared_ptr<ByteBuffer> data_buffer_;
        vector<shared_ptr<ByteBuffer>> dgram_list_;
    };
} // namespace diffraflow

#endif