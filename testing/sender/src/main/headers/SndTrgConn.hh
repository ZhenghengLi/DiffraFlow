#ifndef __SndTrgConn_H__
#define __SndTrgConn_H__

#include "GenericConnection.hh"
#include <log4cxx/logger.h>

namespace diffraflow {

    class SndDatTran;

    class SndTrgConn : public GenericConnection {
    public:
        SndTrgConn(int sock_fd, SndDatTran* dat_tran);
        ~SndTrgConn();

    protected:
        ProcessRes process_payload_(const char* payload_buffer, const size_t payload_size) override;

    private:
        SndDatTran* data_transfer_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif