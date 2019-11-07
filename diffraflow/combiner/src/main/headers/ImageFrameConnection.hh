#ifndef ImageFrameConnection_H
#define ImageFrameConnection_H

#include <iostream>
#include <atomic>

#include "Decoder.hh"

using std::atomic;

namespace shine {

    class ImageCache;

    class ImageFrameConnection {
    private:
        char* buffer_;
        size_t buffer_size_;
        size_t slice_begin_;
        char* pkt_data_;
        size_t pkt_maxlen_;
        int client_sock_fd_;
        ImageCache* image_cache_;
        atomic<bool> done_flag_;

    private:
        bool start_connection_();
        bool transferring_();
        void shift_left_(const size_t position, const size_t limit);

    public:
        ImageFrameConnection(int sock_fd, ImageCache* img_cache_);
        ~ImageFrameConnection();

        void run();
        bool done();
        void set_stop() {
            done_flag_ = true;
        }

    };
}

#endif