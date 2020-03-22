#ifndef CmbImgFrmConn_H
#define CmbImgFrmConn_H

#include "GenericConnection.hh"
#include <log4cxx/logger.h>
#include <atomic>

using std::atomic;
using std::atomic_bool;
using std::atomic_uint;
using std::atomic_ulong;

namespace diffraflow {

    class CmbImgCache;

    class CmbImgFrmConn: public GenericConnection {
    public:
        CmbImgFrmConn(int sock_fd, CmbImgCache* img_cache_);
        ~CmbImgFrmConn();

    public:
        struct {
            atomic_ulong total_processed_frame_size;
            atomic_ulong total_processed_frame_counts;
        } frame_metrics;

        struct {
            // for calculating compression ratio
            atomic_ulong total_compressed_size;
            atomic_ulong total_uncompressed_size;
        } compression_metrics;

        json::value collect_metrics() override;

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
