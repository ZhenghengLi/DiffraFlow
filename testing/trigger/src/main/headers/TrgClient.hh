#ifndef __TrgClient_H__
#define __TrgClient_H__

#include <log4cxx/logger.h>

#include "GenericClient.hh"

namespace diffraflow {
    class TrgClient : public GenericClient {
    public:
        TrgClient(string sender_host, int sender_port, uint32_t trigger_id);
        ~TrgClient();

        bool trigger();
        void set_event_index(uint32_t event_index);
        uint32_t get_event_index();
        uint32_t next_event_index();

    private:
        uint32_t current_event_index_;
        char* send_buffer_;
        char* recv_buffer_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif