#ifndef GenericClient_H
#define GenericClient_H

#include <string>
#include <log4cxx/logger.h>

using std::string;

namespace diffraflow {
    class GenericClient {
    public:
        GenericClient(string hostname, int port, uint32_t id,
            uint32_t greet_hd, uint32_t send_hd);
        ~GenericClient();

        bool connect_to_server();
        void close_connection();
        int send_payload(const char* payload_buffer, const uint32_t payload_type, size_t buffer_size);
        int recv_payload(char* payload_buffer, uint32_t& payload_type, size_t& buffer_size);

    protected:
        string      dest_host_;
        int         dest_port_;
        uint32_t    client_id_;
        int         client_sock_fd_;

        uint32_t    greeting_head_;
        uint32_t    sending_head_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif