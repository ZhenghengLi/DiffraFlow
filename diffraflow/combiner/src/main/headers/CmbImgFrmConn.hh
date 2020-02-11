#ifndef CmbImgFrmConn_H
#define CmbImgFrmConn_H

#include "GenericConnection.hh"
#include <log4cxx/logger.h>

namespace diffraflow {

    class CmbImgCache;

    class CmbImgFrmConn: public GenericConnection {
    public:
        CmbImgFrmConn(int sock_fd, CmbImgCache* img_cache_);
        ~CmbImgFrmConn();

    protected:
        ProcessRes process_payload_(const size_t payload_position,
            const uint32_t payload_size, const uint32_t payload_type);

    private:
        CmbImgCache* image_cache_;

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif
