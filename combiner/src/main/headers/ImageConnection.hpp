#ifndef ImageConnection_H
#define ImageConnection_H

#include <iostream>
#include <atomic>

using std::atomic;

namespace shine {

    class ImageCacheServer;

    class ImageConnection {
    private:
        char* buffer_;
        size_t buffer_size_;
        int client_sock_fd_;
        ImageCacheServer* image_cache_server_;
        atomic<bool> done_flag_;

    public:
        ImageConnection(int sock_fd, ImageCacheServer* server);
        ~ImageConnection();

        void run();
        bool done();

    };
}

#endif