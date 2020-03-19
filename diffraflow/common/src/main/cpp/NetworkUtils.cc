#include "NetworkUtils.hh"
#include "PrimitiveSerializer.hh"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>

bool diffraflow::NetworkUtils::send_packet(
    int            client_sock_fd,
    uint32_t       packet_head,
    const char*    payload_head_buffer,
    const size_t   payload_head_size,
    const char*    payload_data_buffer,
    const size_t   payload_data_size,
    log4cxx::LoggerPtr logger) {

    // send packet head
    char head_buffer[8];
    gPS.serializeValue<uint32_t>(packet_head, head_buffer, 4);
    gPS.serializeValue<uint32_t>(payload_head_size + payload_data_size, head_buffer + 4, 4);
    for (size_t pos = 0; pos < 8;) {
        int count = write(client_sock_fd, head_buffer + pos, 8 - pos);
        if (count < 0) {
            LOG4CXX_WARN(logger, "error found when sending data: " << strerror(errno));
            return false;
        } else {
            pos += count;
        }
    }
    LOG4CXX_DEBUG(logger, "done a write for packet head.");

    // send payload head
    for (size_t pos = 0; pos < payload_head_size;) {
        int count = write(client_sock_fd, payload_head_buffer + pos, payload_head_size - pos);
        if (count < 0) {
            LOG4CXX_WARN(logger, "error found when sending data: " << strerror(errno));
            return false;
        } else {
            pos += count;
        }
    }
    LOG4CXX_DEBUG(logger, "done a write for payload head.");

    // send payload data
    for (size_t pos = 0; pos < payload_data_size;) {
        int count = write(client_sock_fd, payload_data_buffer + pos, payload_data_size - pos);
        if (count < 0) {
            LOG4CXX_WARN(logger, "error found when sending data: " << strerror(errno));
            return false;
        } else {
            pos += count;
        }
    }
    LOG4CXX_DEBUG(logger, "done a write for payload data.");

    return true;
}