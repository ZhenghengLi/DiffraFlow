#include "TrgClient.hh"
#include "PrimitiveSerializer.hh"
#include "Decoder.hh"

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

log4cxx::LoggerPtr diffraflow::TrgClient::logger_ = log4cxx::Logger::getLogger("TrgClient");

diffraflow::TrgClient::TrgClient(string sender_host, int sender_port, uint32_t trigger_id)
    : GenericClient(sender_host, sender_port, trigger_id, 0xBBFF1234, 0xBBB22FFF, 0xFFF22BBB) {
    send_buffer_ = new char[4];
    recv_buffer_ = new char[4];
}

diffraflow::TrgClient::~TrgClient() {
    delete[] send_buffer_;
    delete[] recv_buffer_;
}

bool diffraflow::TrgClient::trigger(const uint32_t event_index) {
    if (not_connected()) {
        LOG4CXX_INFO(logger_, "connection to sender is lost, try to reconnect.");
        if (connect_to_server()) {
            LOG4CXX_INFO(logger_, "reconnected to sender.");
        } else {
            LOG4CXX_WARN(logger_, "failed to reconnect to sender.");
            return false;
        }
    }

    gPS.serializeValue<uint32_t>(event_index, send_buffer_, 4);

    if (send_one_(send_buffer_, 4, nullptr, 0)) {
        size_t payload_size = 0;
        if (receive_one_(recv_buffer_, 4, payload_size)) {
            if (payload_size != 4) {
                LOG4CXX_ERROR(logger_, "got unexpected payload_size: " << payload_size);
                close_connection();
                return false;
            }
            uint32_t response_code = gDC.decode_byte<uint32_t>(recv_buffer_, 0, 3);
            if (response_code == 0) {
                LOG4CXX_INFO(logger_, "successfully sent event " << event_index);
                return true;
            } else {
                LOG4CXX_WARN(logger_, "failed to send event " << event_index);
                return false;
            }
        } else {
            LOG4CXX_ERROR(logger_, "failed to receive response code.");
            close_connection();
            return false;
        }
    } else {
        LOG4CXX_ERROR(logger_, "failed to send event index");
        close_connection();
        return false;
    }
}
