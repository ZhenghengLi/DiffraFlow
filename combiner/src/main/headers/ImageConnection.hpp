#ifndef ImageConnection_H
#define ImageConnection_H

#include <iostream>
#include <atomic>

using std::atomic;

namespace shine {

    class ImageCache;

    class ImageConnection {
    private:
        char* buffer_;
        size_t buffer_size_;
        int client_sock_fd_;
        ImageCache* image_cache_;
        atomic<bool> done_flag_;

    public:
        ImageConnection(int sock_fd, ImageCache* img_cache_);
        ~ImageConnection();

        void run();
        bool done();

        bool start_connection();
        void transfering();

    };
}

#endif