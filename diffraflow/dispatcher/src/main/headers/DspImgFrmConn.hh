#ifndef DspImgFrmConn_H
#define DspImgFrmConn_H

#include <cstddef>
#include <cstdint>

#include "GenericConnection.hh"

namespace diffraflow {

    class DspSender;

    class DspImgFrmConn: public GenericConnection {
    public:
        DspImgFrmConn(int sock_fd, DspSender** sender_arr, size_t sender_cnt);
        ~DspImgFrmConn();

    protected:
        void before_transferring_();
        bool do_transferring_();

    private:
        int hash_long_(int64_t value);

    private:
        DspSender** sender_array_;
        size_t      sender_count_;
    };
}

#endif
