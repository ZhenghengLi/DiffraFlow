#include "NetworkUtils.hh"
#include "PrimitiveSerializer.hh"
#include "Decoder.hh"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>

bool diffraflow::NetworkUtils::send_packet(const int client_sock_fd, const uint32_t packet_head,
    const char* payload_head_buffer, const size_t payload_head_size, const char* payload_data_buffer,
    const size_t payload_data_size, log4cxx::LoggerPtr logger) {

    if (client_sock_fd < 0) {
        LOG4CXX_ERROR(logger, "invalid client_sock_fd");
        return false;
    }

    if (payload_head_buffer == nullptr && payload_data_buffer == nullptr) {
        LOG4CXX_ERROR(logger, "cannot only send head without payload.");
        return false;
    }

    // serialize packet head
    char head_buffer[8];
    gPS.serializeValue<uint32_t>(packet_head, head_buffer, 4);
    uint32_t packet_size = 0;
    if (payload_head_buffer != nullptr) {
        packet_size += payload_head_size;
    }
    if (payload_data_buffer != nullptr) {
        packet_size += payload_data_size;
    }
    gPS.serializeValue<uint32_t>(packet_size, head_buffer + 4, 4);

    // send packet head
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
    if (payload_head_buffer != nullptr) {
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
    }

    // send payload data
    if (payload_data_buffer != nullptr) {
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
    }

    return true;
}

bool diffraflow::NetworkUtils::receive_packet(const int client_sock_fd, const uint32_t packet_head, char* buffer,
    const size_t buffer_size, size_t& packet_size, log4cxx::LoggerPtr logger) {

    if (client_sock_fd < 0) {
        LOG4CXX_ERROR(logger, "invalid client_sock_fd");
        return false;
    }

    // read packet head and size
    char head_size_buffer[8];
    for (size_t pos = 0; pos < 8;) {
        int count = read(client_sock_fd, head_size_buffer + pos, 8 - pos);
        if (count < 0) {
            LOG4CXX_WARN(logger, "error found when receiving data: " << strerror(errno));
            return false;
        } else if (count == 0) {
            LOG4CXX_INFO(logger, "socket " << client_sock_fd << " is closed.");
            return false;
        } else {
            pos += count;
        }
    }
    // extract packet head and packet size
    uint32_t pkt_head = gDC.decode_byte<uint32_t>(head_size_buffer, 0, 3);
    uint32_t pkt_size = gDC.decode_byte<uint32_t>(head_size_buffer, 4, 7);
    // head and size check for packet
    if (pkt_head != packet_head) {
        LOG4CXX_INFO(logger, "got wrong packet, close the connection.");
        return false;
    }
    if (pkt_size > buffer_size) {
        LOG4CXX_INFO(logger, "got too long packet, close the connection.");
        return false;
    }
    if (pkt_size < 4) {
        LOG4CXX_INFO(logger, "got too short packet, close the connection.");
        return false;
    }
    // read payload data
    for (size_t pos = 0; pos < pkt_size;) {
        int count = read(client_sock_fd, buffer + pos, pkt_size - pos);
        if (count < 0) {
            LOG4CXX_WARN(logger, "error found when receiving data: " << strerror(errno));
            return false;
        } else if (count == 0) {
            LOG4CXX_INFO(logger, "socket " << client_sock_fd << " is closed.");
            return false;
        } else {
            pos += count;
        }
    }
    // set packet size
    packet_size = pkt_size;

    return true;
}