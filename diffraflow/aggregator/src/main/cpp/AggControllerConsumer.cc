#include "AggControllerConsumer.hh"
#include "AggMetrics.hh"

log4cxx::LoggerPtr diffraflow::AggControllerConsumer::logger_ = log4cxx::Logger::getLogger("AggControllerConsumer");

diffraflow::AggControllerConsumer::AggControllerConsumer(AggMetrics* metrics) : AggBaseConsumer("controller-consumer") {
    aggregated_metrics_ = metrics;
}

diffraflow::AggControllerConsumer::~AggControllerConsumer() {
    //
}

void diffraflow::AggControllerConsumer::process_message_(const pulsar::Message& message) {
    string message_data = message.getDataAsString();
    if (message_data.empty()) {
        return;
    }
    std::error_code err;
    string metrics_key = message.getPartitionKey();
    json::value metrics_json_full = json::value::parse(message_data, err);
    if (err) {
        LOG4CXX_WARN(logger_, "error found when parsing json: " << err.message());
    } else {
        json::value metrics_json_simplified = simplify_metrics_(metrics_json_full);
        aggregated_metrics_->set_metrics("controller", metrics_key, metrics_json_simplified);
    }
}

json::value diffraflow::AggControllerConsumer::simplify_metrics_(const json::value& metrics_json) {
    // currently just return the original one
    return metrics_json;
}