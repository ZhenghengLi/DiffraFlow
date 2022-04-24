#include "GenericClient.hh"
#include "PrimitiveSerializer.hh"
#include "NetworkUtils.hh"

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#include <string.h>

log4cxx::LoggerPtr diffraflow::GenericClient::logger_ = log4cxx::Logger::getLogger("GenericClient");

diffraflow::GenericClient::GenericClient(
    string hostname, int port, uint32_t id, uint32_t greet_hd, uint32_t send_hd, uint32_t recv_hd) {
    is_ipc_ = false;
    dest_sock_path_ = "";
    dest_host_ = hostname;
    dest_port_ = port;

    client_id_ = id;
    greeting_head_ = greet_hd;
    sending_head_ = send_hd;
    receiving_head_ = recv_hd;
    client_sock_fd_ = -1;
    client_port_ = -1;

    tcp_keepalive_ = 1;
    tcp_keepidle_ = 60;
    tcp_keepintvl_ = 30;
    tcp_keepcnt_ = 9;

    init_metrics_();
}

diffraflow::GenericClient::GenericClient(
    string sock_path, uint32_t id, uint32_t greet_hd, uint32_t send_hd, uint32_t recv_hd) {
    is_ipc_ = true;
    dest_sock_path_ = sock_path;
    dest_host_ = "";
    dest_port_ = -1;

    client_id_ = id;
    greeting_head_ = greet_hd;
    sending_head_ = send_hd;
    receiving_head_ = recv_hd;
    client_sock_fd_ = -1;

    tcp_keepalive_ = 1;
    tcp_keepidle_ = 60;
    tcp_keepintvl_ = 30;
    tcp_keepcnt_ = 9;

    init_metrics_();
}

diffraflow::GenericClient::~GenericClient() { close_connection(); }

void diffraflow::GenericClient::set_tcp_keep_pramas(int alive, int idle, int intvl, int cnt) {
    tcp_keepalive_ = alive;
    tcp_keepidle_ = idle;
    tcp_keepintvl_ = intvl;
    tcp_keepcnt_ = cnt;
}

void diffraflow::GenericClient::init_metrics_() {
    network_metrics.connected = false;
    network_metrics.connecting_action_counts = 0;
    network_metrics.total_sent_size = 0;
    network_metrics.total_sent_counts = 0;
    network_metrics.total_received_size = 0;
    network_metrics.total_received_counts = 0;
}

void diffraflow::GenericClient::set_client_port(int port) {
    if (port > 0 && port < 65536) {
        client_port_ = port;
    } else {
        client_port_ = -1;
    }
}

bool diffraflow::GenericClient::connect_to_server() {

    network_metrics.connecting_action_counts++;

    if (is_ipc_) {
        if (!connect_to_server_ipc_()) {
            LOG4CXX_ERROR(logger_, "Failed to connect to server running on sock file " << dest_sock_path_);
            return false;
        }
    } else {
        if (!connect_to_server_tcp_()) {
            LOG4CXX_ERROR(logger_, "Failed to connect to server running on " << dest_host_ << ":" << dest_port_);
            return false;
        }
    }

    if (greet_to_server_()) {
        if (is_ipc_) {
            LOG4CXX_INFO(logger_, "Successfully connected to server running on sock file " << dest_sock_path_);
        } else {
            LOG4CXX_INFO(logger_, "Successfully connected to server running on " << dest_host_ << ":" << dest_port_);
        }
        network_metrics.connected = true;
        return true;
    } else {
        LOG4CXX_ERROR(logger_, "Failed to greet to server, close the connection.");
        close_connection();
        return false;
    }
}

bool diffraflow::GenericClient::connect_to_server_tcp_() {
    if (client_sock_fd_ >= 0) {
        LOG4CXX_WARN(logger_, "connection to server is already set up.");
        return true;
    }
    addrinfo hints, *infoptr;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    int result = getaddrinfo(dest_host_.c_str(), NULL, &hints, &infoptr);
    if (result) {
        LOG4CXX_ERROR(logger_, "getaddrinfo: " << gai_strerror(result));
        return false;
    }
    client_sock_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (client_sock_fd_ < 0) {
        LOG4CXX_ERROR(logger_, "Socket creationg error: " << strerror(errno));
        freeaddrinfo(infoptr);
        return false;
    }
    if (!NetworkUtils::enable_tcp_keepalive(
            client_sock_fd_, tcp_keepalive_, tcp_keepidle_, tcp_keepintvl_, tcp_keepcnt_, logger_)) {
        close_connection();
        LOG4CXX_ERROR(logger_, "found error when setting tcp keepalive on socket " << client_sock_fd_);
        freeaddrinfo(infoptr);
        return false;
    }
    if (client_port_ > 0) {
        sockaddr_in client_addr;
        memset(&client_addr, 0, sizeof(client_addr));
        client_addr.sin_family = AF_INET;
        client_addr.sin_addr.s_addr = INADDR_ANY;
        client_addr.sin_port = htons(client_port_);
        if (bind(client_sock_fd_, (sockaddr*)&client_addr, sizeof(client_addr)) < 0) {
            LOG4CXX_ERROR(logger_, "bind: " << strerror(errno));
            freeaddrinfo(infoptr);
            return false;
        }
    }
    ((sockaddr_in*)(infoptr->ai_addr))->sin_port = htons(dest_port_);
    if (connect(client_sock_fd_, infoptr->ai_addr, infoptr->ai_addrlen)) {
        close_connection();
        LOG4CXX_ERROR(logger_, "Connection to " << dest_host_ << ":" << dest_port_ << " failed: " << strerror(errno));
        freeaddrinfo(infoptr);
        return false;
    } else {
        freeaddrinfo(infoptr);
        return true;
    }
}

bool diffraflow::GenericClient::connect_to_server_ipc_() {
    if (client_sock_fd_ >= 0) {
        LOG4CXX_WARN(logger_, "connection to server is already set up.");
        return true;
    }
    if (dest_sock_path_.length() > 100) {
        LOG4CXX_ERROR(logger_, "server sock path is too long: " << dest_sock_path_);
        return false;
    }
    struct stat stat_buffer;
    if (stat(dest_sock_path_.c_str(), &stat_buffer) != 0) {
        LOG4CXX_ERROR(logger_, "sock file " << dest_sock_path_ << " does not exist.");
        return false;
    }
    sockaddr_un server_addr;
    client_sock_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (client_sock_fd_ < 0) {
        LOG4CXX_ERROR(logger_, "Socket creationg error: " << strerror(errno));
        return false;
    }
    bzero(&server_addr, sizeof(server_addr));
    server_addr.sun_family = AF_UNIX;
    strcpy(server_addr.sun_path, dest_sock_path_.c_str());
    if (connect(client_sock_fd_, (struct sockaddr*)&server_addr, sizeof(server_addr))) {
        close_connection();
        LOG4CXX_ERROR(logger_, "Connection to sock " << dest_sock_path_ << " failed: " << strerror(errno));
        return false;
    } else {
        return true;
    }
}

bool diffraflow::GenericClient::greet_to_server_() {
    if (client_sock_fd_ < 0) {
        LOG4CXX_ERROR(logger_, "initial connection to server is not set up.");
        return false;
    }
    // send greeting message for varification
    char buffer[12];
    gPS.serializeValue<uint32_t>(greeting_head_, buffer, 4);
    gPS.serializeValue<uint32_t>(4, buffer + 4, 4);
    gPS.serializeValue<uint32_t>(client_id_, buffer + 8, 4);
    for (size_t pos = 0; pos < 12;) {
        int count = write(client_sock_fd_, buffer + pos, 12 - pos);
        if (count > 0) {
            pos += count;
        } else {
            LOG4CXX_ERROR(logger_, "error found when doing the first write: " << strerror(errno));
            return false;
        }
    }
    for (size_t pos = 0; pos < 4;) {
        int count = read(client_sock_fd_, buffer + pos, 4 - pos);
        if (count > 0) {
            pos += count;
        } else {
            LOG4CXX_ERROR(logger_, "error found when doing the first read: " << strerror(errno));
            return false;
        }
    }
    int response_code = 0;
    gPS.deserializeValue<int32_t>(&response_code, buffer, 4);
    if (response_code != 1234) {
        LOG4CXX_ERROR(logger_, "Got wrong response code, close the connection.");
        return false;
    } else {
        return true;
    }
}

void diffraflow::GenericClient::close_connection() {
    if (client_sock_fd_ >= 0) {
        shutdown(client_sock_fd_, SHUT_RDWR);
        close(client_sock_fd_);
        client_sock_fd_ = -1;
    }
    network_metrics.connected = false;
}

bool diffraflow::GenericClient::not_connected() { return (client_sock_fd_ < 0); }

string diffraflow::GenericClient::get_server_address() {
    if (is_ipc_) {
        return dest_sock_path_;
    } else {
        return dest_host_ + ":" + std::to_string(dest_port_);
    }
}

bool diffraflow::GenericClient::send_one_(const char* payload_head_buffer, const size_t payload_head_size,
    const char* payload_data_buffer, const size_t payload_data_size) {

    if (NetworkUtils::send_packet(client_sock_fd_, sending_head_, payload_head_buffer, payload_head_size,
            payload_data_buffer, payload_data_size, logger_)) {

        network_metrics.total_sent_size += (8 + payload_head_size + payload_data_size);
        // 8 is the size of packet head
        network_metrics.total_sent_counts += 1;

        return true;
    } else {
        return false;
    }
}

bool diffraflow::GenericClient::send_head_(const uint32_t packet_size) {
    if (NetworkUtils::send_packet_head(client_sock_fd_, sending_head_, packet_size, logger_)) {

        network_metrics.total_sent_size += 8;
        network_metrics.total_sent_counts += 1;

        return true;
    } else {
        return false;
    }
}

bool diffraflow::GenericClient::send_segment_(const char* segment_data_buffer, const size_t segment_data_size) {
    if (NetworkUtils::send_packet_segment(client_sock_fd_, segment_data_buffer, segment_data_size, logger_)) {

        network_metrics.total_sent_size += segment_data_size;

        return true;
    } else {
        return false;
    }
}

bool diffraflow::GenericClient::receive_one_(char* buffer, const size_t buffer_size, size_t& payload_size) {

    if (NetworkUtils::receive_packet(client_sock_fd_, receiving_head_, buffer, buffer_size, payload_size, logger_)) {

        network_metrics.total_received_size += 8 + payload_size;
        // 8 is the size of packet head
        network_metrics.total_received_counts += 1;

        return true;
    } else {
        return false;
    }
}

bool diffraflow::GenericClient::receive_one_(
    uint32_t& payload_type, shared_ptr<ByteBuffer>& payload_data, const uint32_t max_payload_size) {
    if (NetworkUtils::receive_packet(
            client_sock_fd_, receiving_head_, payload_type, payload_data, logger_, max_payload_size)) {
        network_metrics.total_received_size += 12 + payload_data->size();
        // 12 is the size of packet head and payload type
        network_metrics.total_received_counts += 1;
        return true;
    } else {
        return false;
    }
}

json::value diffraflow::GenericClient::collect_metrics() {

    json::value network_metrics_json;
    network_metrics_json["connected"] = json::value::boolean(network_metrics.connected.load());
    network_metrics_json["connecting_action_counts"] =
        json::value::number(network_metrics.connecting_action_counts.load());
    network_metrics_json["total_sent_size"] = json::value::number(network_metrics.total_sent_size.load());
    network_metrics_json["total_sent_counts"] = json::value::number(network_metrics.total_sent_counts.load());
    network_metrics_json["total_received_size"] = json::value::number(network_metrics.total_received_size.load());
    network_metrics_json["total_received_counts"] = json::value::number(network_metrics.total_received_counts.load());

    json::value root_json;
    root_json["network_stats"] = network_metrics_json;

    return root_json;
}