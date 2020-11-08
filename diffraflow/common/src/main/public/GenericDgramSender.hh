#ifndef __GenericDgramSender_H__
#define __GenericDgramSender_H__

#include <string>
#include <arpa/inet.h>
#include <log4cxx/logger.h>
#include <atomic>

#include "MetricsProvider.hh"

using std::string;
using std::atomic;

namespace diffraflow {
    class GenericDgramSender : public MetricsProvider {
    public:
        explicit GenericDgramSender(int sndbufsize = 4 * 1024 * 1024);
        virtual ~GenericDgramSender();

        bool init_addr_sock(string host, int port);
        string get_receiver_address();
        bool send_datagram(const char* buffer, size_t len);
        void close_sock();

    public:
        struct {
            atomic<uint64_t> total_send_count;
            atomic<uint64_t> total_send_size;
            atomic<uint64_t> total_succ_count;
            atomic<uint64_t> total_succ_size;
            atomic<uint64_t> total_error_count;
            atomic<uint64_t> total_zero_count;
            atomic<uint64_t> total_partial_count;
        } dgram_metrics;

        virtual json::value collect_metrics() override;

    protected:
        string receiver_sock_host_;
        int receiver_sock_port_;

        int sender_sock_fd_;
        int sender_sock_bs_;

        struct sockaddr_in receiver_addr_;
        socklen_t receiver_addr_len_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif