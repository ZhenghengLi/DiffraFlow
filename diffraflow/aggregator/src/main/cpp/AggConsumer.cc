#include "AggConsumer.hh"
#include "AggMetrics.hh"
#include <cpprest/json.h>

using namespace web;
using std::unique_lock;
using std::lock_guard;

log4cxx::LoggerPtr diffraflow::AggConsumer::logger_ = log4cxx::Logger::getLogger("AggConsumer");

diffraflow::AggConsumer::AggConsumer(AggMetrics* metrics, const string name, const string topic) {
    aggregated_metrics_ = metrics;
    consumer_name_ = name;
    consumer_topic_ = topic;
    consumer_thread_ = nullptr;
    consumer_status_ = kNotStart;
}

diffraflow::AggConsumer::~AggConsumer() { stop(); }

bool diffraflow::AggConsumer::start(int timeoutMs) {

    lock_guard<mutex> lg(op_mtx_);

    if (consumer_status_ == kRunning || consumer_status_ == kStopping) {
        LOG4CXX_WARN(logger_, "consumer " << consumer_name_ << " is already started or is stopping.");
        return false;
    }
    consumer_status_ = kNotStart;
    consumer_thread_ = new thread([this, timeoutMs]() {
        pulsar::Consumer consumer;
        pulsar::Result result = aggregated_metrics_->pulsar_client_->subscribe(
            consumer_topic_, aggregated_metrics_->subscription_name_, consumer);
        if (result == pulsar::ResultOk) {
            LOG4CXX_INFO(logger_, "successfully subscribed " << consumer_topic_);
            consumer_status_ = kRunning;
            consumer_cv_.notify_all();

        } else {
            LOG4CXX_WARN(logger_, "failed to subscribe " << consumer_topic_);
            consumer_status_ = kStopped;
            consumer_cv_.notify_all();
            return;
        }
        while (consumer_status_ == kRunning) {
            pulsar::Message message;
            pulsar::Result recv_result = consumer.receive(message, timeoutMs);
            if (recv_result == pulsar::ResultOk) {
                LOG4CXX_DEBUG(logger_, "received one message from topic " << message.getTopicName() << " with key "
                                                                          << message.getPartitionKey() << ".");
                process_message_(message);
                consumer.acknowledge(message);
            }
        }
        result = consumer.unsubscribe();
        if (result == pulsar::ResultOk) {
            LOG4CXX_INFO(logger_, "successfully unsubscribed topic " << consumer_topic_);
        } else {
            LOG4CXX_WARN(logger_, "failed to unsubscribe topic " << consumer_topic_);
        }
        consumer_status_ = kStopped;
        consumer_cv_.notify_all();
    });
    // wait start result
    unique_lock<mutex> ulk(consumer_mtx_);
    consumer_cv_.wait(ulk, [this]() { return consumer_status_ != kNotStart; });
    return consumer_status_ == kRunning;
}

void diffraflow::AggConsumer::stopping() {

    lock_guard<mutex> lg(op_mtx_);

    if (consumer_status_ == kRunning) {
        consumer_status_ = kStopping;
    }
}

void diffraflow::AggConsumer::stop() {

    lock_guard<mutex> lg(op_mtx_);

    if (consumer_status_ == kNotStart) {
        return;
    }

    if (consumer_thread_ != nullptr) {
        consumer_status_ = kStopping;
        consumer_thread_->join();
        delete consumer_thread_;
        consumer_thread_ = nullptr;
        consumer_status_ = kStopped;
        consumer_cv_.notify_all();
    }
}

void diffraflow::AggConsumer::wait() {
    unique_lock<mutex> ulk(consumer_mtx_);
    consumer_cv_.wait(ulk, [this]() { return consumer_status_ != kRunning; });
}

void diffraflow::AggConsumer::process_message_(const pulsar::Message& message) {
    string message_data = message.getDataAsString();
    if (message_data.empty()) {
        return;
    }
    std::error_code err;
    string metrics_key = message.getPartitionKey();
    json::value metrics_json = json::value::parse(message_data, err);
    if (err) {
        LOG4CXX_WARN(logger_, "error found when parsing json: " << err.message());
    } else {
        aggregated_metrics_->set_metrics_(consumer_name_, metrics_key, metrics_json);
    }
}
