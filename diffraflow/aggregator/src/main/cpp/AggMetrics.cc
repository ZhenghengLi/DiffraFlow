#include "AggMetrics.hh"
#include "AggSenderConsumer.hh"
#include "AggDispatcherConsumer.hh"
#include "AggCombinerConsumer.hh"
#include "AggIngesterConsumer.hh"
#include "AggMonitorConsumer.hh"

#include <chrono>
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <pulsar/ClientConfiguration.h>

using std::lock_guard;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::system_clock;
using std::milli;

log4cxx::LoggerPtr diffraflow::AggMetrics::logger_ = log4cxx::Logger::getLogger("AggMetrics");

diffraflow::AggMetrics::AggMetrics(string pulsar_url, int threads_count) {
    pulsar::ClientConfiguration client_config;
    if (threads_count > 0) {
        client_config.setIOThreads(threads_count);
    }
    pulsar_client_ = new pulsar::Client(pulsar_url, client_config);

    sender_consumer_ = nullptr;
    dispatcher_consumer_ = nullptr;
    combiner_consumer_ = nullptr;
    ingester_consumer_ = nullptr;
    monitor_consumer_ = nullptr;
}

diffraflow::AggMetrics::~AggMetrics() {

    stop_all();

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

    duration<double, milli> current_time = system_clock::now().time_since_epoch();
    metrics_json_["update_timestamp"] = json::value::number(current_time.count());
    metrics_json_["update_timestamp_unit"] = json::value::string("milliseconds");
}

json::value diffraflow::AggMetrics::get_metrics() {
    lock_guard<mutex> lg(metrics_json_mtx_);
    return metrics_json_;
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

void diffraflow::AggMetrics::stopping_sender_consumer() {
    if (sender_consumer_ != nullptr) {
        sender_consumer_->stopping();
    }
}

void diffraflow::AggMetrics::stop_sender_consumer() {
    if (sender_consumer_ != nullptr) {
        sender_consumer_->stop();
        delete sender_consumer_;
        sender_consumer_ = nullptr;
    }
}

bool diffraflow::AggMetrics::start_dispatcher_consumer(const string topic, int timeoutMs) {
    if (dispatcher_consumer_ != nullptr) {
        return false;
    }
    dispatcher_consumer_ = new AggDispatcherConsumer(this);
    if (dispatcher_consumer_->start(pulsar_client_, topic, timeoutMs)) {
        return true;
    } else {
        dispatcher_consumer_->stop();
        delete dispatcher_consumer_;
        dispatcher_consumer_ = nullptr;
    }
}

void diffraflow::AggMetrics::stopping_dispatcher_consumer() {
    if (dispatcher_consumer_ != nullptr) {
        dispatcher_consumer_->stopping();
    }
}

void diffraflow::AggMetrics::stop_dispatcher_consumer() {
    if (dispatcher_consumer_ != nullptr) {
        dispatcher_consumer_->stop();
        delete dispatcher_consumer_;
        dispatcher_consumer_ = nullptr;
    }
}

bool diffraflow::AggMetrics::start_combiner_consumer(const string topic, int timeoutMs) {
    if (combiner_consumer_ != nullptr) {
        return false;
    }
    combiner_consumer_ = new AggCombinerConsumer(this);
    if (combiner_consumer_->start(pulsar_client_, topic, timeoutMs)) {
        return true;
    } else {
        combiner_consumer_->stop();
        delete combiner_consumer_;
        combiner_consumer_ = nullptr;
    }
}

void diffraflow::AggMetrics::stopping_combiner_consumer() {
    if (combiner_consumer_ != nullptr) {
        combiner_consumer_->stopping();
    }
}

void diffraflow::AggMetrics::stop_combiner_consumer() {
    if (combiner_consumer_ != nullptr) {
        combiner_consumer_->stop();
        delete combiner_consumer_;
        combiner_consumer_ = nullptr;
    }
}

bool diffraflow::AggMetrics::start_ingester_consumer(const string topic, int timeoutMs) {
    if (ingester_consumer_ != nullptr) {
        return false;
    }
    ingester_consumer_ = new AggIngesterConsumer(this);
    if (ingester_consumer_->start(pulsar_client_, topic, timeoutMs)) {
        return true;
    } else {
        ingester_consumer_->stop();
        delete ingester_consumer_;
        ingester_consumer_ = nullptr;
    }
}

void diffraflow::AggMetrics::stopping_ingester_consumer() {
    if (ingester_consumer_ != nullptr) {
        ingester_consumer_->stopping();
    }
}

void diffraflow::AggMetrics::stop_ingester_consumer() {
    if (ingester_consumer_ != nullptr) {
        ingester_consumer_->stop();
        delete ingester_consumer_;
        ingester_consumer_ = nullptr;
    }
}

bool diffraflow::AggMetrics::start_monitor_consumer(const string topic, int timeoutMs) {
    if (monitor_consumer_ != nullptr) {
        return false;
    }
    monitor_consumer_ = new AggMonitorConsumer(this);
    if (monitor_consumer_->start(pulsar_client_, topic, timeoutMs)) {
        return true;
    } else {
        monitor_consumer_->stop();
        delete monitor_consumer_;
        monitor_consumer_ = nullptr;
    }
}

void diffraflow::AggMetrics::stopping_monitor_consumer() {
    if (monitor_consumer_ != nullptr) {
        monitor_consumer_->stopping();
    }
}

void diffraflow::AggMetrics::stop_monitor_consumer() {
    if (monitor_consumer_ != nullptr) {
        monitor_consumer_->stop();
        delete monitor_consumer_;
        monitor_consumer_ = nullptr;
    }
}

void diffraflow::AggMetrics::wait_all() {
    if (sender_consumer_ != nullptr) {
        sender_consumer_->wait();
    }
    if (dispatcher_consumer_ != nullptr) {
        dispatcher_consumer_->wait();
    }
    if (combiner_consumer_ != nullptr) {
        combiner_consumer_->wait();
    }
    if (ingester_consumer_ != nullptr) {
        ingester_consumer_->wait();
    }
    if (monitor_consumer_ != nullptr) {
        monitor_consumer_->wait();
    }
}

void diffraflow::AggMetrics::stop_all() {

    stopping_sender_consumer();
    stopping_dispatcher_consumer();
    stopping_combiner_consumer();
    stopping_ingester_consumer();
    stopping_monitor_consumer();

    stop_sender_consumer();
    stop_dispatcher_consumer();
    stop_combiner_consumer();
    stop_ingester_consumer();
    stop_monitor_consumer();
}