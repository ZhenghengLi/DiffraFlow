#include "MetricsReporter.hh"

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

using namespace web;
using namespace http;
using namespace experimental::listener;
using std::unique_lock;
using std::lock_guard;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::system_clock;
using std::milli;

log4cxx::LoggerPtr diffraflow::MetricsReporter::logger_
    = log4cxx::Logger::getLogger("MetricsReporter");

diffraflow::MetricsReporter::MetricsReporter() {
    listener_ = nullptr;
    pulsar_client_ = nullptr;
    pulsar_producer_ = nullptr;
    sender_thread_ = nullptr;
    sender_is_running_ = false;
}

diffraflow::MetricsReporter::~MetricsReporter() {
    stop_http_server();
    stop_msg_producer();
}

void diffraflow::MetricsReporter::add(string name, MetricsProvider* mp_obj) {
    metrics_scalar_[name] = mp_obj;
}

void diffraflow::MetricsReporter::add(string name, vector<MetricsProvider*>& mp_obj_vec) {
    metrics_array_[name] = mp_obj_vec;
}

void diffraflow::MetricsReporter::add(string name, MetricsProvider** mp_obj_arr, size_t len) {
    vector<MetricsProvider*> mp_obj_vec(mp_obj_arr, mp_obj_arr + len);
    metrics_array_[name] = mp_obj_vec;
}

void diffraflow::MetricsReporter::clear() {
    metrics_scalar_.clear();
    metrics_array_.clear();
}

json::value diffraflow::MetricsReporter::aggregate_metrics_() {
    json::value root_json;

    duration<double, milli> current_time = system_clock::now().time_since_epoch();
    root_json["timestamp"] = json::value::number( (uint64_t) current_time.count() );
    root_json["timestamp_unit"] = json::value::string("milliseconds");

    for (map<string, MetricsProvider*>::iterator iter = metrics_scalar_.begin();
        iter != metrics_scalar_.end(); ++iter) {
        root_json[iter->first] = iter->second->collect_metrics();
    }
    for (map<string, vector<MetricsProvider*> >::iterator iter = metrics_array_.begin();
        iter != metrics_array_.end(); ++iter) {
        for (size_t i = 0 ; i < iter->second.size(); i++) {
            root_json[iter->first][i] = iter->second[i]->collect_metrics();
        }
    }

    return root_json;
}

bool diffraflow::MetricsReporter::start_msg_producer(
    const char* broker_address, const char* topic, const char* msg_key, size_t report_period) {
    if (pulsar_client_ != nullptr) {
        LOG4CXX_WARN(logger_, "pulsar client has already been started.");
        return false;
    }

    pulsar_client_ = new pulsar::Client(broker_address);

    pulsar::ProducerConfiguration producer_config;
    producer_config.setPartitionsRoutingMode(pulsar::ProducerConfiguration::RoundRobinDistribution);
    pulsar::Result result = pulsar_client_->createProducer(topic, producer_config, *pulsar_producer_);
    if (result != pulsar::ResultOk) {
        LOG4CXX_ERROR(logger_, "Error found when creating producer: " << result);
        return false;
    }

    message_key_ = msg_key;
    report_period_ = report_period;
    sender_is_running_ = true;
    sender_thread_ = new thread(
        [this] () {
            while (sender_is_running_) {
                unique_lock<mutex> ulk(wait_mtx_);
                if (!sender_is_running_) break;

                json::value metrics_json = aggregate_metrics_();
                pulsar::Message message = pulsar::MessageBuilder().
                    setPartitionKey(message_key_).
                    setContent(metrics_json.serialize()).
                    build();
                pulsar::Result result = pulsar_producer_->send(message);
                if (result == pulsar::ResultOk) {
                    LOG4CXX_DEBUG(logger_, "successfully sent metrics with timestamp: " << metrics_json["timestamp"]);
                } else {
                    LOG4CXX_DEBUG(logger_, "failed to send metrics with timestamp: " << metrics_json["timestamp"]);
                }

                wait_cv_.wait_for(ulk, milliseconds(report_period_));
            }
        }
    );

    return true;
}

void diffraflow::MetricsReporter::stop_msg_producer() {
    if (pulsar_client_ == nullptr) return;

    {   // make sure stop the thread when it is in wait state
        lock_guard<mutex> lg(wait_mtx_);
        sender_is_running_ = false;
        wait_cv_.notify_all();
    }

    sender_thread_->join();
    delete sender_thread_;
    sender_thread_ = nullptr;
    pulsar::Result result = pulsar_client_->close();
    if (result != pulsar::ResultOk) {
        LOG4CXX_WARN(logger_, "Error found when close pulsar client: " << result);
    }
    delete pulsar_client_;
    pulsar_client_ = nullptr;
    delete pulsar_producer_;
    pulsar_producer_ = nullptr;
}

bool diffraflow::MetricsReporter::start_http_server(const char* host, int port) {
    if (listener_ != nullptr) {
        LOG4CXX_WARN(logger_, "http server has already been started.");
        return false;
    }
    uri_builder uri_b;
    uri_b.set_scheme("http");
    uri_b.set_host(host);
    uri_b.set_port(port);
    listener_ = new http_listener(uri_b.to_uri());
    listener_->support(methods::GET, std::bind(&MetricsReporter::handleGet_, this, std::placeholders::_1));

    try {
        listener_->open().wait();
        return true;
    } catch (std::exception& e) {
        LOG4CXX_ERROR(logger_, "failed to start http server: " << e.what());
        return false;
    } catch(...) {
        LOG4CXX_ERROR(logger_, "failed to start http server with unknown error.");
        return false;
    }

}

void diffraflow::MetricsReporter::handleGet_(http_request message) {
    http_response response;
    utility::string_t relative_path = uri::decode(message.relative_uri().path());
    if (relative_path == "/") {
        json::value paths_json;
        paths_json[0] = json::value::string("/stat");
        json::value root_json;
        root_json["paths"] = paths_json;
        message.reply(status_codes::OK, root_json);
    } else if (relative_path == "/stat") {
        message.reply(status_codes::OK, aggregate_metrics_());
    } else {
        message.reply(status_codes::NotFound);
    }
}

void diffraflow::MetricsReporter::stop_http_server() {
    if (listener_ == nullptr) return;

    try {
        listener_->close().wait();
    } catch (std::exception& e) {
        LOG4CXX_WARN(logger_, "exception found when closing http listener: " << e.what());
    } catch (...) {
        LOG4CXX_WARN(logger_, "an unknown exception found when closing http listener.");
    }

    delete listener_;
    listener_ = nullptr;
}