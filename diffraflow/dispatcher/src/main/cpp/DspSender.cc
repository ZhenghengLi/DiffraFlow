#include "DspSender.hh"
#include "PrimitiveSerializer.hh"

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

log4cxx::LoggerPtr diffraflow::DspSender::logger_ = log4cxx::Logger::getLogger("DspSender");

diffraflow::DspSender::DspSender(string hostname, int port, int id)
    : GenericClient(hostname, port, id, 0xDDCC1234, 0xDDD22CCC, 0xCCC22DDD) {}

diffraflow::DspSender::~DspSender() {}

bool diffraflow::DspSender::send(const char* data, const size_t len) {
    lock_guard<mutex> lg(mtx_send_);
    // try to connect if lose connection
    if (not_connected()) {
        if (connect_to_server()) {
            LOG4CXX_INFO(logger_, "reconnected to combiner.");
        } else {
            LOG4CXX_WARN(logger_, "failed to reconnect to combiner, discard data in buffer.");
            return false;
        }
    }
    if (send_one_(data, len, nullptr, 0)) {
        return true;
    } else {
        close_connection();
        return false;
    }
}
