#ifndef DspImgFrmConn_H
#define DspImgFrmConn_H

#include <cstddef>
#include <cstdint>

#include "GenericConnection.hh"
#include <log4cxx/logger.h>

namespace diffraflow {

    class DspSender;

    class DspImgFrmConn : public GenericConnection {
    public:
        DspImgFrmConn(int sock_fd, DspSender** sender_arr, size_t sender_cnt);
        ~DspImgFrmConn();

    protected:
        bool do_receiving_and_processing_() override;

    private:
        uint32_t hash_long_(uint64_t value);

    private:
        DspSender** sender_array_;
        size_t sender_count_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
