#ifndef DspImgFrmConn_H
#define DspImgFrmConn_H

#include <cstddef>
#include <cstdint>

#include "GenericConnection.hh"
#include <log4cxx/logger.h>

namespace diffraflow {

    class DspSender;

    class DspImgFrmConn: public GenericConnection {
    public:
        DspImgFrmConn(int sock_fd, DspSender** sender_arr, size_t sender_cnt);
        ~DspImgFrmConn();

    protected:
        bool process_payload_(
            const char*  payload_buffer,
            const size_t payload_size) override;

    private:
        int hash_long_(int64_t value);

    private:
        DspSender** sender_array_;
        size_t      sender_count_;

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif
