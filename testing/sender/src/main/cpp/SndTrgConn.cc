#include "SndTrgConn.hh"
#include "SndDatTran.hh"
#include "Decoder.hh"

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

log4cxx::LoggerPtr diffraflow::SndTrgConn::logger_ = log4cxx::Logger::getLogger("SndTrgConn");

diffraflow::SndTrgConn::SndTrgConn(int sock_fd, SndDatTran* dat_tran)
    : GenericConnection(sock_fd, 0xBBFF1234, 0xBBB22FFF, 0xFFF22BBB, 1024) {
    data_transfer_ = dat_tran;
}

diffraflow::SndTrgConn::~SndTrgConn() {}

diffraflow::GenericConnection::ProcessRes diffraflow::SndTrgConn::process_payload_(
    const char* payload_buffer, const size_t payload_size) {
    // extract payload type
    uint32_t bunch_id = gDC.decode_byte<uint32_t>(payload_buffer, 0, 3);

    return kProcessed;
}