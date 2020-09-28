#include "AggMetrics.hh"
#include "AggSenderConsumer.hh"

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <pulsar/ClientConfiguration.h>

using std::lock_guard;

log4cxx::LoggerPtr diffraflow::AggMetrics::logger_ = log4cxx::Logger::getLogger("AggMetrics");

diffraflow::AggMetrics::AggMetrics(string pulsar_url, int threads_count) {
    pulsar::ClientConfiguration client_config;
    if (threads_count > 0) {
        client_config.setIOThreads(threads_count);
    }
    pulsar_client_ = new pulsar::Client(pulsar_url, client_config);

    sender_consumer_ = nullptr;
}

diffraflow::AggMetrics::~AggMetrics() {
    pulsar_client_->close();
    delete pulsar_client_;
    pulsar_client_ = nullptr;
}

void diffraflow::AggMetrics::set_metrics(const string topic, const string key, const json::value& value) {
    lock_guard<mutex> lg(metrics_json_mtx_);
    if (!metrics_json_.has_field(topic)) {
        metrics_json_[topic] = json::value();
    }
    metrics_json_[topic][key] = value;
}

string diffraflow::AggMetrics::get_metrics() {
    lock_guard<mutex> lg(metrics_json_mtx_);
    return metrics_json_.serialize();
}

bool diffraflow::AggMetrics::start_sender_consumer(const string topic, int timeoutMs) {
    if (sender_consumer_ != nullptr) {
        return false;
    }
    sender_consumer_ = new AggSenderConsumer(this);
    if (sender_consumer_->start(pulsar_client_, topic, timeoutMs)) {
        return true;
    } else {
        sender_consumer_->stop();
        delete sender_consumer_;
        sender_consumer_ = nullptr;
    }
}

void diffraflow::AggMetrics::stop_sender_consumer() {
    if (sender_consumer_ != nullptr) {
        sender_consumer_->stop();
        delete sender_consumer_;
        sender_consumer_ = nullptr;
    }
}
