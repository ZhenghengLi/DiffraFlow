#ifndef ImageCacheServer_H
#define ImageCacheServer_H

namespace shine {
    class ImageCacheServer {
    private:
        int port_;
        int server_sock_fd_;
        bool run_;

    public:
        ImageCacheServer(int p);
        ~ImageCacheServer();

        bool create_sock();
        int accept_client();

        void serve();

        void clean();

    };
}

#endif