#ifndef __TrgClient_H__
#define __TrgClient_H__

#include <log4cxx/logger.h>

#include "GenericClient.hh"

namespace diffraflow {
    class TrgClient : public GenericClient {
    public:
        TrgClient(string sender_host, int sender_port, uint32_t trigger_id);
        ~TrgClient();

        bool trigger(const uint32_t event_index);

    private:
        char* send_buffer_;
        char* recv_buffer_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif