#ifndef CmbImgFrmConn_H
#define CmbImgFrmConn_H

#include "GenericConnection.hh"
#include <log4cxx/logger.h>
#include <atomic>

using std::atomic;

namespace diffraflow {

    class CmbImgCache;

    class CmbImgFrmConn : public GenericConnection {
    public:
        CmbImgFrmConn(int sock_fd, CmbImgCache* img_cache_);
        ~CmbImgFrmConn();

    public:
        struct {
            atomic<uint64_t> total_processed_frame_size;
            atomic<uint64_t> total_processed_frame_counts;
        } frame_metrics;

        json::value collect_metrics() override;

    protected:
        bool do_receiving_and_processing_() override;

    private:
        CmbImgCache* image_cache_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
