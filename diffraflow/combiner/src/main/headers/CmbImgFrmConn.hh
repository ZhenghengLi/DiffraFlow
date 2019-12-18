#ifndef CmbImgFrmConn_H
#define CmbImgFrmConn_H

#include "GeneralConnection.hh"

namespace diffraflow {

    class CmbImgCache;

    class CmbImgFrmConn: public GeneralConnection {
    private:
        CmbImgCache* image_cache_;

    protected:
        void before_transferring_();
        bool do_transferring_();

    public:
        CmbImgFrmConn(int sock_fd, CmbImgCache* img_cache_);
        ~CmbImgFrmConn();

    };
}

#endif
