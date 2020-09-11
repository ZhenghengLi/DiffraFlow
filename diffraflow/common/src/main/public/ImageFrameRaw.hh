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

        bool add_dgram(shared_ptr<vector<char>>& dgram);
        // currently this method is not implemented.
        void sort_dgrams();
        // return total dgrams size if ok, otherwise return -1
        int check_dgrams_integrity();
        // methods for data sending
        shared_ptr<vector<char>>& get_dgram(size_t index);
        size_t get_dgram_count();

    public:
        uint64_t bunch_id; // key
        int16_t module_id; // 0 -- 15

        uint8_t dgram_mod_id;
        uint16_t dgram_frm_sn;
        uint8_t dgram_seg_sn;

    private:
        shared_ptr<vector<char>> data_buffer_;
        vector<shared_ptr<vector<char>>> dgram_list_;
    };
} // namespace diffraflow

#endif