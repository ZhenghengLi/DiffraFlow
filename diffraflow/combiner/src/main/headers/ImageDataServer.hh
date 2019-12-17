#ifndef ImageDataServer_H
#define ImageDataServer_H

#include "GeneralServer.hh"

namespace diffraflow {

    class ImageDataConnection;
    class ImageCache;

    class ImageDataServer: public GeneralServer {
    private:
        ImageCache* image_cache_;

    protected:
        GeneralConnection* new_connection_(int client_sock_fd);

    public:
        ImageDataServer(string sock_path, ImageCache* img_cache);
        ~ImageDataServer();

    };
}

#endif
