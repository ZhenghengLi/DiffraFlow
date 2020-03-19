#ifndef GenericConnection_H
#define GenericConnection_H

#include <iostream>
#include <atomic>
#include <log4cxx/logger.h>

using std::atomic;

namespace diffraflow {
    class GenericConnection {
    public:
        GenericConnection(int sock_fd,
            uint32_t greet_hd, uint32_t recv_hd, uint32_t send_hd,
            size_t buff_sz, size_t pkt_ml );
        virtual ~GenericConnection();

        void run();
        bool done();
        void stop();

    protected:
        uint32_t connection_id_;
        uint32_t greeting_head_;
        uint32_t receiving_head_;
        uint32_t sending_head_;

        char* buffer_;
        size_t buffer_size_;
        size_t slice_begin_;
        size_t pkt_maxlen_;
        int client_sock_fd_;
        atomic<bool> done_flag_;

    protected:
        enum ProcessRes {kContinue, kBreak, kStop};

    protected:
        bool send_one_(
            const char*    payload_head_buffer,
            const size_t   payload_head_size,
            const char*    payload_data_buffer,
            const size_t   payload_data_size);
        // methods to be implemented
        virtual ProcessRes process_payload_(const char* payload_buffer,
            const uint32_t payload_size, const uint32_t payload_type);

    private:
        bool start_connection_();
        void shift_left_(const size_t position, const size_t limit);
        void before_receiving_();
        bool do_receiving_and_processing_();

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif
