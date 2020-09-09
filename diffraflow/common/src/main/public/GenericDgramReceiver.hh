#ifndef __GenericDgramReceiver_H__
#define __GenericDgramReceiver_H__

#include <string>

#include <arpa/inet.h>

using std::string;

namespace diffraflow {
    class GenericDgramReceiver {
    public:
        GenericDgramReceiver(string host, int port);
        ~GenericDgramReceiver();

    protected:
        bool create_udp_sock_();

    protected:
        string receiver_sock_host_;
        int receiver_sock_port_;
        int receiver_sock_fd_;

        struct sockaddr_in receiver_addr_;
        struct sockaddr_in sender_addr_;
    };
} // namespace diffraflow

#endif