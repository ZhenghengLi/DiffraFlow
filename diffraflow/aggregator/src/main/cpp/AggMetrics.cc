#include "AggMetrics.hh"

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

#include <pulsar/ClientConfiguration.h>

log4cxx::LoggerPtr diffraflow::AggMetrics::logger_ = log4cxx::Logger::getLogger("AggMetrics");

diffraflow::AggMetrics::AggMetrics(string pulsar_url, int threads_count) {
    pulsar::ClientConfiguration client_config;
    if (threads_count > 0) {
        client_config.setIOThreads(threads_count);
    }
    pulsar_client_ = new pulsar::Client(pulsar_url, client_config);
}

diffraflow::AggMetrics::~AggMetrics() {
    pulsar_client_->close();
    delete pulsar_client_;
    pulsar_client_ = nullptr;
}
