#ifndef __NetworkUtils_H__
#define __NetworkUtils_H__

#include <memory>
#include <cstdint>
#include <log4cxx/logger.h>

#include "ByteBuffer.hh"

using std::shared_ptr;
using std::make_shared;

namespace diffraflow {
    namespace NetworkUtils {

        bool send_packet(const int client_sock_fd, const uint32_t packet_head, const char* payload_head_buffer,
            const size_t payload_head_size, const char* payload_data_buffer, const size_t payload_data_size,
            log4cxx::LoggerPtr logger);

        bool send_packet_head(const int client_sock_fd, const uint32_t packet_head, const uint32_t packet_size,
            log4cxx::LoggerPtr logger);

        bool send_packet_segment(const int client_sock_fd, const char* segment_data_buffer,
            const size_t segment_data_size, log4cxx::LoggerPtr logger);

        bool receive_packet(const int client_sock_fd, const uint32_t packet_head, char* buffer,
            const size_t buffer_size, size_t& packet_size, log4cxx::LoggerPtr logger);

        bool receive_packet(const int client_sock_fd, const uint32_t packet_head, uint32_t& payload_type,
            shared_ptr<ByteBuffer>& payload_data, log4cxx::LoggerPtr logger, const uint32_t max_payload_size = 1048576);

        bool enable_tcp_keepalive(int sock, int alive, int idle, int intvl, int cnt, log4cxx::LoggerPtr logger);

    }; // namespace NetworkUtils
} // namespace diffraflow

#endif