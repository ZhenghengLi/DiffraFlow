#ifndef DispatchingConnection_H
#define DispatchingConnection_H

#include <cstddef>
#include <cstdint>

#include "GeneralConnection.hh"

namespace diffraflow {

    class Sender;

    class DispatchingConnection: public GeneralConnection {
    public:
        DispatchingConnection(int sock_fd, Sender** sender_arr, size_t sender_cnt);
        ~DispatchingConnection();

    protected:
        void before_transferring_();
        bool do_transferring_();

    private:
        int hash_long_(int64_t value);

    private:
        Sender** sender_array_;
        size_t   sender_count_;
    };
}

#endif