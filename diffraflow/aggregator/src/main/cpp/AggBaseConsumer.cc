#include "AggBaseConsumer.hh"

using std::unique_lock;
using std::lock_guard;

log4cxx::LoggerPtr diffraflow::AggBaseConsumer::logger_ = log4cxx::Logger::getLogger("AggBaseConsumer");

diffraflow::AggBaseConsumer::AggBaseConsumer(string name) {
    consumer_name_ = name;
    consumer_thread_ = nullptr;
    consumer_status_ = kNotStart;
}

diffraflow::AggBaseConsumer::~AggBaseConsumer() { stop(); }

bool diffraflow::AggBaseConsumer::start(pulsar::Client* client, const string topic, int timeoutMs) {

    lock_guard<mutex> lg(op_mtx_);

    if (consumer_status_ == kRunning || consumer_status_ == kStopping) {
        LOG4CXX_WARN(logger_, "consumer " << consumer_name_ << " is already started or is stopping.");
        return false;
    }
    consumer_status_ = kNotStart;
    consumer_thread_ = new thread([this, client, topic, timeoutMs]() {
        pulsar::Consumer consumer;
        pulsar::Result result = client->subscribe(topic, consumer_name_, consumer);
        if (result == pulsar::ResultOk) {
            LOG4CXX_INFO(logger_, "successfully subscribed " << topic);
            consumer_status_ = kRunning;
            consumer_cv_.notify_all();

        } else {
            LOG4CXX_WARN(logger_, "failed to subscribe " << topic);
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
            LOG4CXX_INFO(logger_, "successfully unsubscribed topic " << topic);
        } else {
            LOG4CXX_WARN(logger_, "failed to unsubscribe topic " << topic);
        }
        consumer_status_ = kStopped;
        consumer_cv_.notify_all();
    });
    // wait start result
    unique_lock<mutex> ulk(consumer_mtx_);
    consumer_cv_.wait(ulk, [this]() { return consumer_status_ != kNotStart; });
    return consumer_status_ == kRunning;
}

void diffraflow::AggBaseConsumer::stopping() {

    lock_guard<mutex> lg(op_mtx_);

    if (consumer_status_ == kRunning) {
        consumer_status_ = kStopping;
    }
}

void diffraflow::AggBaseConsumer::stop() {

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

void diffraflow::AggBaseConsumer::wait() {
    unique_lock<mutex> ulk(consumer_mtx_);
    consumer_cv_.wait(ulk, [this]() { return consumer_status_ != kRunning; });
}