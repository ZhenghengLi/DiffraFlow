#ifndef __SndTcpSender_H__
#define __SndTcpSender_H__

#include "GenericClient.hh"
#include <log4cxx/logger.h>

namespace diffraflow {
    class SndTcpSender : public GenericClient {
    public:
        SndTcpSender(string dispatcher_host, int dispatcher_port, uint32_t sender_id, int sender_port = -1);
        ~SndTcpSender();

        bool send_frame(const char* buffer, size_t len);

    private:
        char* head_buffer_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif