#ifndef __SndTrgConn_H__
#define __SndTrgConn_H__

#include "GenericConnection.hh"
#include <log4cxx/logger.h>
#include <atomic>

using std::atomic;

namespace diffraflow {

    class SndDatTran;

    class SndTrgConn : public GenericConnection {
    public:
        SndTrgConn(int sock_fd, SndDatTran* dat_tran);
        ~SndTrgConn();

    public:
        struct {
            atomic<uint64_t> total_succ_push_counts;
            atomic<uint64_t> total_fail_push_counts;
        } push_metrics;

        json::value collect_metrics() override;

    protected:
        ProcessRes process_payload_(const char* payload_buffer, const size_t payload_size) override;

    private:
        SndDatTran* data_transfer_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif