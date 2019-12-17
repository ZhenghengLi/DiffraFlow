#ifndef ImageDataConnection_H
#define ImageDataConnection_H

#include "GeneralConnection.hh"

#define WAIT_TIME_MS 50

namespace diffraflow {

    class ImageCache;

    class ImageDataConnection: public GeneralConnection {
    private:
        ImageCache* image_cache_;

    protected:
        void before_transferring_();
        bool do_transferring_();

    public:
        ImageDataConnection(int sock_fd, ImageCache* img_cache_);
        ~ImageDataConnection();

    };
}

#endif
