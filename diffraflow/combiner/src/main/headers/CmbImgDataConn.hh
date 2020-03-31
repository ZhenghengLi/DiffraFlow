#ifndef CmbImgDataConn_H
#define CmbImgDataConn_H

#include "GenericConnection.hh"
#include <log4cxx/logger.h>
#include <msgpack.hpp>

namespace diffraflow {

    class CmbImgCache;

    class CmbImgDataConn: public GenericConnection {
    public:
        CmbImgDataConn(int sock_fd, CmbImgCache* image_cache, size_t max_req_imgct = 10);
        ~CmbImgDataConn();

    protected:
        ProcessRes process_payload_(const char* payload_buffer,
            const size_t payload_size) override;

    private:
        CmbImgCache* image_cache_;
        msgpack::sbuffer image_buffer_;
        size_t max_req_imgct_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif
