#include "AggSenderConsumer.hh"
#include "AggMetrics.hh"

log4cxx::LoggerPtr diffraflow::AggSenderConsumer::logger_ = log4cxx::Logger::getLogger("AggSenderConsumer");

diffraflow::AggSenderConsumer::AggSenderConsumer(string name, AggMetrics* metrics) : AggBaseConsumer(name) {
    aggregated_metrics_ = metrics;
}

diffraflow::AggSenderConsumer::~AggSenderConsumer() {
    //
}

void diffraflow::AggSenderConsumer::process_message_(const pulsar::Message& message) {
    //
}