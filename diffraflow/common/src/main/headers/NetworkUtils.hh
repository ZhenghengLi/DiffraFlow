#ifndef __NetworkUtils_H__
#define __NetworkUtils_H__

#include <cstdint>
#include <log4cxx/logger.h>

namespace diffraflow {
    namespace NetworkUtils {

        bool send_packet(const int client_sock_fd, const uint32_t packet_head, const char* payload_head_buffer,
            const size_t payload_head_size, const char* payload_data_buffer, const size_t payload_data_size,
            log4cxx::LoggerPtr logger);

        bool receive_packet(const int client_sock_fd, const uint32_t packet_head, char* buffer,
            const size_t buffer_size, size_t& packet_size, log4cxx::LoggerPtr logger);
    }; // namespace NetworkUtils
} // namespace diffraflow

#endif