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
        void shift_left_(const size_t position, const size_t limit);
        int32_t get_connection_id_() {
            return connection_id_;
        }

        // methods to be implemented
        virtual void before_transferring_() = 0;
        virtual bool do_transferring_() = 0;

    public:
        GenericConnection(int sock_fd, size_t buff_sz, size_t pkt_ml, uint32_t greet_hd);
        virtual ~GenericConnection();

        void run();
        bool done();
        void stop();


    };
}

#endif
