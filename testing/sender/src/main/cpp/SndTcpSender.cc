#include "SndTcpSender.hh"
#include "PrimitiveSerializer.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

log4cxx::LoggerPtr diffraflow::SndTcpSender::logger_ = log4cxx::Logger::getLogger("SndTcpSender");

diffraflow::SndTcpSender::SndTcpSender(string dispatcher_host, int dispatcher_port, uint32_t sender_id)
    : GenericClient(dispatcher_host, dispatcher_port, sender_id, 0xFFDD1234, 0xFFF22DDD, 0xDDD22FFF) {
    head_buffer_ = new char[4];
    gPS.serializeValue<uint32_t>(0xABCDFFFF, head_buffer_, 4);
}

diffraflow::SndTcpSender::~SndTcpSender() { delete[] head_buffer_; }