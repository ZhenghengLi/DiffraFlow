#ifndef ImageFrameServer_H
#define ImageFrameServer_H

#include "GeneralServer.hh"

namespace diffraflow {

    class ImageFrameConnection;
    class ImageCache;

    class ImageFrameServer: public GeneralServer {
    private:
        ImageCache* image_cache_;

    protected:
        GeneralConnection* new_connection_(int client_sock_fd);

    public:
        ImageFrameServer(int port, ImageCache* img_cache);
        ~ImageFrameServer();

    };
}

#endif
