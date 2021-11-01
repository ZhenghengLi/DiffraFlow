#include "NetworkUtils.hh"
#include "PrimitiveSerializer.hh"
#include "Decoder.hh"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
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

bool diffraflow::NetworkUtils::send_packet_head(
    const int client_sock_fd, const uint32_t packet_head, const uint32_t packet_size, log4cxx::LoggerPtr logger) {
    if (client_sock_fd < 0) {
        LOG4CXX_ERROR(logger, "invalid client_sock_fd");
        return false;
    }

    // serialize packet head
    char head_buffer[8];
    gPS.serializeValue<uint32_t>(packet_head, head_buffer, 4);
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
    return true;
}

bool diffraflow::NetworkUtils::send_packet_segment(const int client_sock_fd, const char* segment_data_buffer,
    const size_t segment_data_size, log4cxx::LoggerPtr logger) {
    if (client_sock_fd < 0) {
        LOG4CXX_ERROR(logger, "invalid client_sock_fd");
        return false;
    }

    if (segment_data_buffer == nullptr) {
        LOG4CXX_ERROR(logger, "null segment_data_buffer pointer.");
        return false;
    }

    for (size_t pos = 0; pos < segment_data_size;) {
        int count = write(client_sock_fd, segment_data_buffer + pos, segment_data_size - pos);
        if (count < 0) {
            LOG4CXX_WARN(logger, "error found when sending data: " << strerror(errno));
            return false;
        } else {
            pos += count;
        }
    }
    LOG4CXX_DEBUG(logger, "done a write for segment data.");

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

bool diffraflow::NetworkUtils::receive_packet(const int client_sock_fd, const uint32_t packet_head,
    uint32_t& payload_type, shared_ptr<ByteBuffer>& payload_data, log4cxx::LoggerPtr logger,
    const uint32_t max_payload_size) {

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
    if (pkt_size < 4) {
        LOG4CXX_INFO(logger, "got too short packet, close the connection.");
        return false;
    }
    // read payload type
    char payload_type_buffer[4];
    for (size_t pos = 0; pos < 4;) {
        int count = read(client_sock_fd, payload_type_buffer + pos, 4 - pos);
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
    payload_type = gDC.decode_byte<uint32_t>(payload_type_buffer, 0, 3);
    // read payload_data
    uint32_t payload_size = pkt_size - 4;
    if (payload_size > max_payload_size) {
        LOG4CXX_WARN(logger, "payload_size is too large: " << payload_size << ".");
        return false;
    }
    payload_data = make_shared<ByteBuffer>(payload_size);
    for (size_t pos = 0; pos < payload_size;) {
        int count = read(client_sock_fd, payload_data->data() + pos, payload_size - pos);
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

    return true;
}

bool diffraflow::NetworkUtils::enable_tcp_keepalive(
    int sock, int alive, int idle, int intvl, int cnt, log4cxx::LoggerPtr logger) {
    if (sock < 0) {
        LOG4CXX_WARN(logger, "invalid socket fd: " << sock);
        return false;
    }
    if (alive >= 0) {
        if (setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &alive, sizeof(int)) < 0) {
            LOG4CXX_WARN(logger, "error found when setting SO_KEEPALIVE: " << strerror(errno));
            return false;
        } else {
            LOG4CXX_INFO(logger, "successfully set SO_KEEPALIVE(" << alive << ") on socket " << sock);
        }
    }
    if (idle > 0) {
        if (setsockopt(sock, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(int)) < 0) {
            LOG4CXX_WARN(logger, "error found when setting TCP_KEEPIDLE: " << strerror(errno));
            return false;
        } else {
            LOG4CXX_INFO(logger, "successfully set TCP_KEEPIDLE(" << idle << ") on socket " << sock);
        }
    }
    if (intvl > 0) {
        if (setsockopt(sock, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(int)) < 0) {
            LOG4CXX_WARN(logger, "error found when setting TCP_KEEPINTVL: " << strerror(errno));
            return false;
        } else {
            LOG4CXX_INFO(logger, "successfully set TCP_KEEPINTVL(" << intvl << ") on socket " << sock);
        }
    }
    if (cnt > 0) {
        if (setsockopt(sock, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof(int)) < 0) {
            LOG4CXX_WARN(logger, "error found when setting TCP_KEEPCNT: " << strerror(errno));
            return false;
        } else {
            LOG4CXX_INFO(logger, "successfully set TCP_KEEPCNT(" << cnt << ") on socket " << sock);
        }
    }
    return true;
}
