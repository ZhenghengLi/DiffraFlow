#include "MetricsReporter.hh"

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

bool diffraflow::MetricsReporter::start_http_server(const char* host, int port) {

    return true;
}
