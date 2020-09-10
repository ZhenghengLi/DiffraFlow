#ifndef __GenericDgramSender_H__
#define __GenericDgramSender_H__

#include <string>
#include <arpa/inet.h>
#include <log4cxx/logger.h>

using std::string;

namespace diffraflow {
    class GenericDgramSender {
    public:
        GenericDgramSender();
        ~GenericDgramSender();

        bool init_addr_sock(string host, int port);
        bool send_datagram(const char* buffer, size_t len);
        void close_sock();

    protected:
        string receiver_sock_host_;
        int receiver_sock_port_;

        int sender_sock_fd_;

        struct sockaddr_in receiver_addr_;
        socklen_t receiver_addr_len_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif