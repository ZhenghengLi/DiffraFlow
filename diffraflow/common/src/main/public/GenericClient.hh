#ifndef GenericClient_H
#define GenericClient_H

#include <string>
#include <atomic>
#include <log4cxx/logger.h>

#include "MetricsProvider.hh"

using std::string;
using std::atomic;

namespace diffraflow {
    class GenericClient: public MetricsProvider {
    public:
        GenericClient(string hostname, int port, uint32_t id,
            uint32_t greet_hd, uint32_t send_hd, uint32_t recv_hd);
        ~GenericClient();

        bool connect_to_server();
        void close_connection();

    public:
        struct {
            atomic<bool>     connected;
            atomic<uint32_t> connecting_action_counts;
            atomic<uint64_t> total_sent_size;
            atomic<uint64_t> total_sent_counts;
            atomic<uint64_t> total_received_size;
            atomic<uint64_t> total_received_counts;
        } network_metrics;

        virtual json::value collect_metrics() override;

    protected:
        bool send_one_(
            const char*    payload_head_buffer,
            const size_t   payload_head_size,
            const char*    payload_data_buffer,
            const size_t   payload_data_size);

        bool receive_one_(
            char*          buffer,
            const size_t   buffer_size,
            size_t&        payload_size);

    protected:
        string      dest_host_;
        int         dest_port_;
        uint32_t    client_id_;
        int         client_sock_fd_;

        uint32_t    greeting_head_;
        uint32_t    sending_head_;
        uint32_t    receiving_head_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif