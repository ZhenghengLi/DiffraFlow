#include "AggCombinerConsumer.hh"
#include "AggMetrics.hh"

log4cxx::LoggerPtr diffraflow::AggCombinerConsumer::logger_ = log4cxx::Logger::getLogger("AggCombinerConsumer");

diffraflow::AggCombinerConsumer::AggCombinerConsumer(AggMetrics* metrics) : AggBaseConsumer("combiner-consumer") {
    aggregated_metrics_ = metrics;
}

diffraflow::AggCombinerConsumer::~AggCombinerConsumer() {
    //
}

void diffraflow::AggCombinerConsumer::process_message_(const pulsar::Message& message) {
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
        aggregated_metrics_->set_metrics("combiner", metrics_key, metrics_json_simplified);
    }
}

json::value diffraflow::AggCombinerConsumer::simplify_metrics_(const json::value& metrics_json) {
    // currently just return the original one
    return metrics_json;
}