#ifndef CmbImgDataConn_H
#define CmbImgDataConn_H

#include "GenericConnection.hh"

namespace diffraflow {

    class CmbImgCache;

    class CmbImgDataConn: public GenericConnection {
    private:
        CmbImgCache* image_cache_;

    protected:
        void before_transferring_();
        bool do_transferring_();

    public:
        CmbImgDataConn(int sock_fd, CmbImgCache* img_cache_);
        ~CmbImgDataConn();

    };
}

#endif
