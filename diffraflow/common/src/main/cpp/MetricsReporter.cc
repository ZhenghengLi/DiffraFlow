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