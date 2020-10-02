#include "AggMetrics.hh"
#include "AggConsumer.hh"

#include <chrono>
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <pulsar/ClientConfiguration.h>

using std::lock_guard;
using std::unique_lock;
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
}

diffraflow::AggMetrics::~AggMetrics() {

    stop_all();

    pulsar_client_->close();
    delete pulsar_client_;
    pulsar_client_ = nullptr;
}

void diffraflow::AggMetrics::set_metrics_(const string topic, const string key, const json::value& value) {
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

bool diffraflow::AggMetrics::start_consumer(const string name, const string topic, int timeoutMs) {
    lock_guard<mutex> lg(consumer_map_mtx_);
    if (consumer_map_.find(name) != consumer_map_.end()) {
        LOG4CXX_WARN(logger_, "consumer " << name << " has already been started.");
        return false;
    }
    AggConsumer* consumer = new AggConsumer(this, name, topic);
    if (consumer->start(timeoutMs)) {
        consumer_map_[name] = consumer;
        return true;
    } else {
        consumer->stop();
        delete consumer;
        consumer = nullptr;
    }
}

void diffraflow::AggMetrics::stop_consumer(const string name) {
    lock_guard<mutex> lg(consumer_map_mtx_);
    if (consumer_map_.find(name) == consumer_map_.end()) {
        LOG4CXX_WARN(logger_, "consumer " << name << " has not yet been started.");
        return;
    }
    consumer_map_[name]->stop();
    delete consumer_map_[name];
    consumer_map_.erase(name);
    if (consumer_map_.empty()) {
        consumer_map_cv_.notify_all();
    }
}

void diffraflow::AggMetrics::wait() {
    unique_lock<mutex> ulk(consumer_map_mtx_);
    consumer_map_cv_.wait(ulk, [this]() { return consumer_map_.empty(); });
}

void diffraflow::AggMetrics::stop_all() {
    lock_guard<mutex> lg(consumer_map_mtx_);
    for (map<string, AggConsumer*>::iterator iter = consumer_map_.begin(); iter != consumer_map_.end(); ++iter) {
        LOG4CXX_INFO(logger_, "stopping consumer " << iter->first << " ...");
        iter->second->stopping();
    }
    for (map<string, AggConsumer*>::iterator iter = consumer_map_.begin(); iter != consumer_map_.end(); ++iter) {
        iter->second->stop();
        delete iter->second;
        LOG4CXX_INFO(logger_, "consumer " << iter->first << " is stopped.");
    }
    consumer_map_.clear();
    consumer_map_cv_.notify_all();
}
