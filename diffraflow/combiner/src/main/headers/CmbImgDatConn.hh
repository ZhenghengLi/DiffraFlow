#ifndef CmbImgDatConn_H
#define CmbImgDatConn_H

#include "GenericConnection.hh"
#include <log4cxx/logger.h>
#include <msgpack.hpp>

namespace diffraflow {

    class CmbImgCache;

    class CmbImgDatConn : public GenericConnection {
    public:
        CmbImgDatConn(int sock_fd, CmbImgCache* image_cache, size_t max_req_imgct = 10);
        ~CmbImgDatConn();

    public:
        struct {
            atomic<uint64_t> total_sent_images;
        } image_metrics;

        json::value collect_metrics() override;

    protected:
        ProcessRes process_payload_(const char* payload_buffer, const size_t payload_size) override;
        bool do_preparing_and_sending_() override;

    private:
        CmbImgCache* image_cache_;
        msgpack::sbuffer image_buffer_;
        size_t max_req_imgct_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
