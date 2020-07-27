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

    send_metrics.total_succ_send_counts = 0;
    send_metrics.total_fail_send_counts = 0;
}

diffraflow::SndTrgConn::~SndTrgConn() {}

diffraflow::GenericConnection::ProcessRes diffraflow::SndTrgConn::process_payload_(
    const char* payload_buffer, const size_t payload_size) {
    // the whole payload is event index
    uint32_t event_index = gDC.decode_byte<uint32_t>(payload_buffer, 0, 3);

    LOG4CXX_DEBUG(logger_, "sending event " << event_index << " ...");
    if (data_transfer_->read_and_send(event_index)) {
        LOG4CXX_DEBUG(logger_, "successfully sent event " << event_index);
        send_metrics.total_succ_send_counts++;
        return kProcessed;
    } else {
        LOG4CXX_WARN(logger_, "failed to send event " << event_index);
        send_metrics.total_fail_send_counts++;
        return kSkipped;
    }
}

json::value diffraflow::SndTrgConn::collect_metrics() {

    json::value root_json = GenericConnection::collect_metrics();

    json::value send_metrics_json;
    send_metrics_json["total_succ_send_counts"] = json::value::number(send_metrics.total_succ_send_counts.load());
    send_metrics_json["total_fail_send_counts"] = json::value::number(send_metrics.total_fail_send_counts.load());

    root_json["send_stats"] = send_metrics_json;

    return root_json;
}