#ifndef ImageFrameConnection_H
#define ImageFrameConnection_H

#include "GeneralConnection.hh"

namespace shine {

    class ImageCache;

    class ImageFrameConnection: public GeneralConnection {
    private:
        ImageCache* image_cache_;

    protected:
        void before_transferring_();
        bool do_transferring_();

    public:
        ImageFrameConnection(int sock_fd, ImageCache* img_cache_);
        ~ImageFrameConnection();

    };
}

#endif