#include "MetricsReporter.hh"

#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>

using namespace web;
using namespace http;
using namespace experimental::listener;

log4cxx::LoggerPtr diffraflow::MetricsReporter::logger_
    = log4cxx::Logger::getLogger("MetricsReporter");

diffraflow::MetricsReporter::MetricsReporter() {

}

diffraflow::MetricsReporter::~MetricsReporter() {

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

}

bool diffraflow::MetricsReporter::start_msg_producer(const char* broker_address, const char* topic) {

    return true;
}

void diffraflow::MetricsReporter::stop_msg_producer() {

}

bool diffraflow::MetricsReporter::start_http_server(const char* host, int port) {
    uri_builder uri_b;
    uri_b.set_scheme("http");
    uri_b.set_host(host);
    uri_b.set_port(port);
    listener_ = http_listener(uri_b.to_uri());
    listener_.support(methods::GET, std::bind(&MetricsReporter::handleGet_, this, std::placeholders::_1));

    try {
        listener_.open().wait();
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

}

void diffraflow::MetricsReporter::stop_http_server() {
    try {
        listener_.close().wait();
    } catch (std::exception& e) {
        LOG4CXX_WARN(logger_, "exception found when closing http listener: " << e.what());
    } catch (...) {
        LOG4CXX_WARN(logger_, "an unknown exception found when closing http listener.");
    }
}