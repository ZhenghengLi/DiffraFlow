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

    push_metrics.total_succ_push_counts = 0;
    push_metrics.total_fail_push_counts = 0;
}

diffraflow::SndTrgConn::~SndTrgConn() {}

diffraflow::GenericConnection::ProcessRes diffraflow::SndTrgConn::process_payload_(
    const char* payload_buffer, const size_t payload_size) {
    // the whole payload is event index
    uint32_t event_index = gDC.decode_byte<uint32_t>(payload_buffer, 0, 3);

    LOG4CXX_DEBUG(logger_, "pushing event " << event_index << " ...");
    if (data_transfer_->push_event(event_index)) {
        LOG4CXX_DEBUG(logger_, "successfully sent event " << event_index);
        push_metrics.total_succ_push_counts++;
        return kProcessed;
    } else {
        LOG4CXX_WARN(logger_, "failed to send event " << event_index);
        push_metrics.total_fail_push_counts++;
        return kSkipped;
    }
}

json::value diffraflow::SndTrgConn::collect_metrics() {

    json::value root_json = GenericConnection::collect_metrics();

    json::value push_metrics_json;
    push_metrics_json["total_succ_push_counts"] = json::value::number(push_metrics.total_succ_push_counts.load());
    push_metrics_json["total_fail_push_counts"] = json::value::number(push_metrics.total_fail_push_counts.load());

    root_json["push_stats"] = push_metrics_json;

    return root_json;
}