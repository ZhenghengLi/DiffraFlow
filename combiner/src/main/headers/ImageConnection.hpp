#ifndef ImageConnection_H
#define ImageConnection_H

#include <iostream>
#include <atomic>

#include "Decoder.hpp"

using std::atomic;

namespace shine {

    class ImageCache;

    class ImageConnection: private Decoder {
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

    public:
        ImageConnection(int sock_fd, ImageCache* img_cache_);
        ~ImageConnection();

        void run();
        bool done();
        void set_stop() {
            done_flag_ = false;
        }


    };
}

#endif