#include "SndTcpSender.hh"
#include "PrimitiveSerializer.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

log4cxx::LoggerPtr diffraflow::SndTcpSender::logger_ = log4cxx::Logger::getLogger("SndTcpSender");

diffraflow::SndTcpSender::SndTcpSender(string dispatcher_host, int dispatcher_port, uint32_t sender_id, int sender_port)
    : GenericClient(dispatcher_host, dispatcher_port, sender_id, 0xFFDD1234, 0xFFF22DDD, 0xDDD22FFF) {
    set_client_port(sender_port);
    head_buffer_ = new char[4];
    gPS.serializeValue<uint32_t>(0xABCDFFFF, head_buffer_, 4);
}

diffraflow::SndTcpSender::~SndTcpSender() {
    delete[] head_buffer_;
    head_buffer_ = nullptr;
}

bool diffraflow::SndTcpSender::send_frame(const char* buffer, size_t len) {
    // try to connect if lose connection
    if (not_connected()) {
        if (connect_to_server()) {
            LOG4CXX_INFO(logger_, "reconnected to dispatcher.");
        } else {
            LOG4CXX_WARN(logger_, "failed to reconnect to dispatcher.");
            return false;
        }
    }
    // send image frame
    if (send_one_(head_buffer_, 4, buffer, len)) {
        LOG4CXX_DEBUG(logger_, "successfully send one image frame.");
        return true;
    } else {
        close_connection();
        LOG4CXX_WARN(logger_, "failed to send one image frame.");
        return false;
    }
}