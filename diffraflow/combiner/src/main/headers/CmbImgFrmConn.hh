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
        ProcessRes process_payload_(const char* payload_buffer,
            const size_t payload_size) override;

    private:
        CmbImgCache* image_cache_;
        char*  buffer_uncompress_;
        size_t buffer_uncompress_limit_;

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif
