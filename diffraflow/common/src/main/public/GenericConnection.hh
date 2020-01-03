#ifndef GenericConnection_H
#define GenericConnection_H

#include <iostream>
#include <atomic>

using std::atomic;

namespace diffraflow {
    class GenericConnection {
    private:
        uint32_t greeting_head_;
        int32_t connection_id_;

    private:
        bool start_connection_();

    protected:
        char* buffer_;
        size_t buffer_size_;
        size_t slice_begin_;
        size_t pkt_maxlen_;
        int client_sock_fd_;
        atomic<bool> done_flag_;

    protected:
        enum ProcessRes {kContinue, kBreak, kStop};

    protected:
        void shift_left_(const size_t position, const size_t limit);
        int32_t get_connection_id_() {
            return connection_id_;
        }

        // methods to be implemented
        virtual void before_transferring_();
        virtual bool do_transferring_();
        virtual ProcessRes process_payload_(const size_t payload_position,
            const uint32_t payload_size, const uint32_t payload_type);

    public:
        GenericConnection(int sock_fd, size_t buff_sz, size_t pkt_ml, uint32_t greet_hd);
        virtual ~GenericConnection();

        void run();
        bool done();
        void stop();


    };
}

#endif
