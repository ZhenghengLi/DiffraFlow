#include "SndTrgConn.hh"
#include "SndDatTran.hh"
#include "Decoder.hh"
#include "PrimitiveSerializer.hh"

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

log4cxx::LoggerPtr diffraflow::SndTrgConn::logger_ = log4cxx::Logger::getLogger("SndTrgConn");

diffraflow::SndTrgConn::SndTrgConn(int sock_fd, SndDatTran* dat_tran)
    : GenericConnection(sock_fd, 0xBBFF1234, 0xBBB22FFF, 0xFFF22BBB, 1024) {
    data_transfer_ = dat_tran;
    succ_res_buff_ = new char[4];
    gPS.serializeValue<uint32_t>(0, succ_res_buff_, 4);
    fail_res_buff_ = new char[4];
    gPS.serializeValue<uint32_t>(1, fail_res_buff_, 4);
}

diffraflow::SndTrgConn::~SndTrgConn() {
    delete succ_res_buff_;
    delete fail_res_buff_;
}

diffraflow::GenericConnection::ProcessRes diffraflow::SndTrgConn::process_payload_(
    const char* payload_buffer, const size_t payload_size) {
    // the whole payload is event index
    uint32_t event_index = gDC.decode_byte<uint32_t>(payload_buffer, 0, 3);
    if (data_transfer_->read_and_send(event_index)) {
        if (send_one_(succ_res_buff_, 4, nullptr, 0)) {
            return kProcessed;
        } else {
            return kFailed;
        }
    } else {
        if (send_one_(fail_res_buff_, 4, nullptr, 0)) {
            return kSkipped;
        } else {
            return kFailed;
        }
    }
}