#include "IngConfig.hh"
#include <log4cxx/logger.h>
#include <log4cxx/ndc.h>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <sstream>
#include <ctime>
#include <cstdlib>
#include <thread>
#include <regex>

#include <cuda_runtime.h>

using std::cout;
using std::endl;
using std::lock_guard;
using std::regex;
using std::regex_match;
using std::regex_replace;

log4cxx::LoggerPtr diffraflow::IngConfig::logger_ = log4cxx::Logger::getLogger("IngConfig");

diffraflow::IngConfig::IngConfig() {
    ingester_id = 0;
    combiner_port = -1;
    combiner_host = "localhost";
    node_name = "NODENAME";
    image_http_port = -1;
    image_http_host = "localhost";
    recnxn_wait_time = 0;
    recnxn_max_count = 0;
    raw_queue_capacity = 100;
    calib_queue_capacity = 100;
    feature_queue_capacity = 100;
    write_queue_capacity = 1000;
    save_calib_data = false;
    save_raw_data = false;

    gpu_enable = false;
    gpu_device_index = -1;

    hdf5_chunk_size = 1;
    hdf5_compress_level = 0;
    hdf5_swmr_mode = true;
    file_imgcnt_limit = 1000;
    file_imgcnt_rand = 0;

    zookeeper_setting_ready_flag_ = false;

    // initial values of dynamic configurations
    dy_run_number_ = 0;
    dy_param_int_ = 20;
    dy_param_double_ = 100;
    dy_param_string_ = "xfel";

    metrics_pulsar_report_period = 1000;
    metrics_http_port = -1;
}

diffraflow::IngConfig::~IngConfig() {}

bool diffraflow::IngConfig::load(const char* filename) {
    list<pair<string, string>> conf_KV_list;
    if (!read_conf_KV_list_(filename, conf_KV_list)) {
        LOG4CXX_ERROR(logger_, "Failed to read configuration file: " << filename);
        return false;
    }
    // parse
    if (zookeeper_parse_setting_(conf_KV_list)) {
        zookeeper_setting_ready_flag_ = true;
    } else {
        LOG4CXX_WARN(logger_, "zookeeper setting is not ready, configuration will not be dynamically updated.")
    }
    map<string, string> dy_conf_map;
    for (list<pair<string, string>>::iterator iter = conf_KV_list.begin(); iter != conf_KV_list.end(); ++iter) {
        string key = iter->first;
        string value = iter->second;
        // for static parameters
        if (key == "ingester_id") {
            ingester_id = atoi(value.c_str());
        } else if (key == "storage_dir") {
            storage_dir = value;
        } else if (key == "save_calib_data") {
            std::istringstream(value) >> std::boolalpha >> save_calib_data;
        } else if (key == "save_raw_data") {
            std::istringstream(value) >> std::boolalpha >> save_raw_data;
        } else if (key == "hdf5_chunk_size") {
            hdf5_chunk_size = atoi(value.c_str());
        } else if (key == "hdf5_compress_level") {
            hdf5_compress_level = atoi(value.c_str());
        } else if (key == "hdf5_swmr_mode") {
            std::istringstream(value) >> std::boolalpha >> hdf5_swmr_mode;
        } else if (key == "file_imgcnt_limit") {
            file_imgcnt_limit = atoi(value.c_str());
        } else if (key == "file_imgcnt_rand") {
            file_imgcnt_rand = atoi(value.c_str());
        } else if (key == "combiner_host") {
            combiner_host = value;
        } else if (key == "combiner_port") {
            combiner_port = atoi(value.c_str());
        } else if (key == "combiner_sock") {
            combiner_sock = value;
        } else if (key == "image_http_host") {
            image_http_host = value;
        } else if (key == "image_http_port") {
            image_http_port = atoi(value.c_str());
        } else if (key == "recnxn_wait_time") {
            recnxn_wait_time = atoi(value.c_str());
        } else if (key == "recnxn_max_count") {
            recnxn_max_count = atoi(value.c_str());
        } else if (key == "raw_queue_capacity") {
            raw_queue_capacity = atoi(value.c_str());
        } else if (key == "calib_queue_capacity") {
            calib_queue_capacity = atoi(value.c_str());
        } else if (key == "feature_queue_capacity") {
            feature_queue_capacity = atoi(value.c_str());
        } else if (key == "write_queue_capacity") {
            write_queue_capacity = atoi(value.c_str());
        } else if (key == "calib_param_file") {
            calib_param_file = value.c_str();
        } else if (key == "gpu_enable") {
            std::istringstream(value) >> std::boolalpha >> gpu_enable;
        } else if (key == "gpu_device_index") {
            gpu_device_index = atoi(value.c_str());
        } else if (key == "metrics_pulsar_broker_address") {
            metrics_pulsar_broker_address = value;
        } else if (key == "metrics_pulsar_topic_name") {
            metrics_pulsar_topic_name = value;
        } else if (key == "metrics_pulsar_message_key") {
            metrics_pulsar_message_key = value;
        } else if (key == "metrics_pulsar_report_period") {
            metrics_pulsar_report_period = atoi(value.c_str());
        } else if (key == "metrics_http_host") {
            metrics_http_host = value;
        } else if (key == "metrics_http_port") {
            metrics_http_port = atoi(value.c_str());
            // for dynamic parameters
        } else {
            dy_conf_map[key] = value;
        }
    }
    const char* node_name_cstr = getenv("NODE_NAME");
    const char* node_ip_cstr = getenv("NODE_IP");
    // set node name
    if (node_name_cstr != NULL) {
        node_name = boost::to_upper_copy<string>(node_name_cstr);
    }
    // adjust combiner_host if its value is NODE_NAME or NODE_IP
    if (combiner_host == "NODE_NAME" && node_name_cstr != NULL) {
        combiner_host = node_name_cstr;
    } else if (combiner_host == "NODE_IP" && node_ip_cstr != NULL) {
        combiner_host = node_ip_cstr;
    }

    if (node_name_cstr != nullptr && regex_match(metrics_pulsar_message_key, regex(".*NODE_NAME.*"))) {
        metrics_pulsar_message_key = regex_replace(metrics_pulsar_message_key, regex("NODE_NAME"), node_name_cstr);
    }

    if (storage_dir.empty()) {
        LOG4CXX_WARN(logger_, "storage_dir is not set, data will not be saved.");
    }

    // correction
    if (metrics_pulsar_report_period < 500) {
        LOG4CXX_WARN(logger_, "pulsar_report_period < 500, use 500 instead.");
        metrics_pulsar_report_period = 500;
    }
    if (hdf5_chunk_size < 1) {
        LOG4CXX_WARN(logger_, "hdf5_chunk_size is too small (< 1), use 1 instead.");
        hdf5_chunk_size = 1;
    }
    if (hdf5_compress_level > 9) {
        LOG4CXX_WARN(logger_, "hdf5_compress_level is too high (> 9), use 9 instead.");
        hdf5_compress_level = 9;
    }
    if (file_imgcnt_limit < 2) {
        LOG4CXX_WARN(logger_, "file_imgcnt_limit is too mall (< 2), use 2 instead.");
        file_imgcnt_limit = 2;
    }
    // validation check for static parameters
    bool succ_flag = true;
    if (ingester_id < 0) {
        LOG4CXX_ERROR(logger_, "invalid ingester_id: " << ingester_id);
        succ_flag = false;
    }
    if (combiner_sock.length() > 100) {
        LOG4CXX_ERROR(logger_, "combiner_sock is too long.");
        succ_flag = false;
    }
    if (combiner_sock.empty() && combiner_port < 0) {
        LOG4CXX_ERROR(logger_, "combiner_sock or combiner_port is not set.");
        succ_flag = false;
    }
    if (image_http_port < 0) {
        LOG4CXX_ERROR(logger_, "invalid image_http_port: " << image_http_port);
        succ_flag = false;
    }
    if (raw_queue_capacity < 1 || raw_queue_capacity > 10000) {
        LOG4CXX_ERROR(logger_, "raw_queue_capacity is out of range " << 1 << "-" << 10000);
        succ_flag = false;
    }
    if (calib_queue_capacity < 1 || calib_queue_capacity > 10000) {
        LOG4CXX_ERROR(logger_, "calib_queue_capacity is out of range " << 1 << "-" << 10000);
        succ_flag = false;
    }
    if (feature_queue_capacity < 1 || feature_queue_capacity > 10000) {
        LOG4CXX_ERROR(logger_, "feature_queue_capacity is out of range " << 1 << "-" << 10000);
        succ_flag = false;
    }
    if (write_queue_capacity < 1 || write_queue_capacity > 10000) {
        LOG4CXX_ERROR(logger_, "write_queue_capacity is out of range " << 1 << "-" << 10000);
        succ_flag = false;
    }
    if (gpu_enable && gpu_device_index >= 0) {
        int device_count = 0;
        cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
        if (cuda_err == cudaSuccess) {
            if (gpu_device_index >= device_count) {
                LOG4CXX_ERROR(
                    logger_, "gpu_device_index " << gpu_device_index << " is out of range [0," << device_count << ")");
                succ_flag = false;
            }
        } else {
            LOG4CXX_ERROR(logger_, "Failed to get gpu device count with error: " << cudaGetErrorString(cuda_err));
            succ_flag = false;
        }
    }
    // check and commit for dynamic parameters
    if (!check_and_commit_(dy_conf_map, time(NULL))) {
        LOG4CXX_ERROR(logger_, "dynamic configurations have invalid values.");
        succ_flag = false;
    }
    if (succ_flag) {

        static_config_json_["storage_dir"] = json::value::string(storage_dir);
        static_config_json_["save_calib_data"] = json::value::boolean(save_calib_data);
        static_config_json_["save_raw_data"] = json::value::boolean(save_raw_data);
        static_config_json_["node_name"] = json::value::string(node_name);
        static_config_json_["ingester_id"] = json::value::number(ingester_id);
        static_config_json_["hdf5_chunk_size"] = json::value::number(hdf5_chunk_size);
        static_config_json_["hdf5_compress_level"] = json::value::number(hdf5_compress_level);
        static_config_json_["hdf5_swmr_mode"] = json::value::number(hdf5_swmr_mode);
        static_config_json_["file_imgcnt_limit"] = json::value::number(file_imgcnt_limit);
        static_config_json_["file_imgcnt_rand"] = json::value::number(file_imgcnt_rand);
        static_config_json_["combiner_host"] = json::value::string(combiner_host);
        static_config_json_["combiner_port"] = json::value::number(combiner_port);
        static_config_json_["combiner_sock"] = json::value::string(combiner_sock);
        static_config_json_["image_http_host"] = json::value::string(image_http_host);
        static_config_json_["image_http_port"] = json::value::number(image_http_port);
        static_config_json_["recnxn_wait_time"] = json::value::number((uint32_t)recnxn_wait_time);
        static_config_json_["recnxn_max_count"] = json::value::number((uint32_t)recnxn_max_count);
        static_config_json_["raw_queue_capacity"] = json::value::number((uint32_t)raw_queue_capacity);
        static_config_json_["calib_queue_capacity"] = json::value::number((uint32_t)calib_queue_capacity);
        static_config_json_["feature_queue_capacity"] = json::value::number((uint32_t)feature_queue_capacity);
        static_config_json_["write_queue_capacity"] = json::value::number((uint32_t)write_queue_capacity);
        static_config_json_["calib_param_file"] = json::value::string(calib_param_file);
        static_config_json_["gpu_enable"] = json::value::boolean(gpu_enable);
        static_config_json_["gpu_device_index"] = json::value::number(gpu_device_index);

        metrics_config_json_["metrics_pulsar_broker_address"] = json::value::string(metrics_pulsar_broker_address);
        metrics_config_json_["metrics_pulsar_topic_name"] = json::value::string(metrics_pulsar_topic_name);
        metrics_config_json_["metrics_pulsar_message_key"] = json::value::string(metrics_pulsar_message_key);
        metrics_config_json_["metrics_pulsar_report_period"] = json::value::number(metrics_pulsar_report_period);
        metrics_config_json_["metrics_http_host"] = json::value::string(metrics_http_host);
        metrics_config_json_["metrics_http_port"] = json::value::number(metrics_http_port);

        return true;
    } else {
        return false;
    }
}

json::value diffraflow::IngConfig::collect_metrics() {
    json::value root_json;

    if (zookeeper_setting_ready_flag_) {
        root_json = DynamicConfiguration::collect_metrics();
    }

    root_json["static_config"] = static_config_json_;

    {
        lock_guard<mutex> dynamic_config_json_lg(dynamic_config_json_mtx_);
        root_json["dynamic_config"] = dynamic_config_json_;
    }

    root_json["metrics_config"] = metrics_config_json_;

    return root_json;
}

bool diffraflow::IngConfig::zookeeper_setting_is_ready() { return zookeeper_setting_ready_flag_; }

void diffraflow::IngConfig::print() {
    // with all locks
    lock_guard<mutex> dy_param_string_lg(dy_param_string_mtx_);

    if (zookeeper_setting_ready_flag_) {
        zookeeper_print_setting();
    }

    cout << "server config:" << endl;
    cout << "- ingester_id = " << ingester_id << endl;
    cout << "- combiner_host = " << combiner_host << endl;
    cout << "- combiner_port = " << combiner_port << endl;
    cout << "- combiner_sock = " << combiner_sock << endl;
    cout << "- image_http_host = " << image_http_host << endl;
    cout << "- image_http_port = " << image_http_port << endl;
    cout << "- recnxn_wait_time = " << recnxn_wait_time << endl;
    cout << "- recnxn_max_count = " << recnxn_max_count << endl;
    cout << "- raw_queue_capacity = " << raw_queue_capacity << endl;
    cout << "- calib_queue_capacity = " << calib_queue_capacity << endl;
    cout << "- feature_queue_capacity = " << feature_queue_capacity << endl;
    cout << "- write_queue_capacity = " << write_queue_capacity << endl;
    cout << "- calib_param_file = " << calib_param_file << endl;
    cout << "- gpu_enable = " << gpu_enable << endl;
    cout << "- gpu_device_index = " << gpu_device_index << endl;
    cout << "- storage_dir = " << storage_dir << endl;
    cout << "- save_calib_data = " << save_calib_data << endl;
    cout << "- save_raw_data = " << save_raw_data << endl;
    cout << "dynamic parameters:" << endl;
    cout << "- dy_param_int = " << dy_param_int_.load() << endl;
    cout << "- dy_param_double = " << dy_param_double_.load() << endl;
    cout << "- dy_param_string = " << dy_param_string_ << endl;
    cout << "metrics config:" << endl;
    cout << "- metrics_pulsar_broker_address = " << metrics_pulsar_broker_address << endl;
    cout << "- metrics_pulsar_topic_name = " << metrics_pulsar_topic_name << endl;
    cout << "- metrics_pulsar_message_key = " << metrics_pulsar_message_key << endl;
    cout << "- metrics_pulsar_report_period = " << metrics_pulsar_report_period << endl;
    cout << "- metrics_http_host = " << metrics_http_host << endl;
    cout << "- metrics_http_port = " << metrics_http_port << endl;
}

bool diffraflow::IngConfig::check_and_commit_(const map<string, string>& conf_map, const time_t conf_mtime) {

    // with all locks
    lock_guard<mutex> dy_param_string_lg(dy_param_string_mtx_);

    // values before commit
    int tmp_dy_run_number = dy_run_number_.load();
    int tmp_dy_param_int = dy_param_int_.load();
    double tmp_dy_param_double = dy_param_double_.load();
    string tmp_dy_param_string = dy_param_string_;

    // convert
    for (map<string, string>::const_iterator iter = conf_map.begin(); iter != conf_map.end(); ++iter) {
        string key = iter->first;
        string value = iter->second;
        if (key == "dy_param_int") {
            tmp_dy_param_int = atoi(value.c_str());
        } else if (key == "dy_run_number") {
            tmp_dy_run_number = atoi(value.c_str());
        } else if (key == "dy_param_double") {
            tmp_dy_param_double = atof(value.c_str());
        } else if (key == "dy_param_string") {
            tmp_dy_param_string = value;
        }
    }

    // validation check
    bool invalid_flag = false;
    if (tmp_dy_param_int < 10) {
        cout << "invalid configuration: dy_param_int(" << tmp_dy_param_int << ") is out of range [10, inf)." << endl;
        invalid_flag = true;
    }
    if (tmp_dy_param_double > 1000) {
        cout << "invalid configuration: dy_param_double(" << tmp_dy_param_double << ") is out of range (-inf, 1000]."
             << endl;
        invalid_flag = true;
    }
    if (tmp_dy_param_string.length() < 2) {
        cout << "invalid configuration: dy_param_string(" << tmp_dy_param_string << ") is too short." << endl;
        invalid_flag = true;
    }
    if (tmp_dy_run_number < 0) {
        cout << "invalid configuration: dy_run_number(" << tmp_dy_run_number << ") is less than zero." << endl;
        invalid_flag = true;
    }

    if (invalid_flag) {
        return false;
    }

    // commit change
    if (dy_param_int_ != tmp_dy_param_int) {
        cout << "configuration changed: dy_param_int [ " << dy_param_int_ << " -> " << tmp_dy_param_int << " ]."
             << endl;
        dy_param_int_ = tmp_dy_param_int;
    }
    if (dy_param_double_ != tmp_dy_param_double) {
        cout << "configuration changed: dy_param_double [ " << dy_param_double_ << " -> " << tmp_dy_param_double
             << " ]." << endl;
        dy_param_double_ = tmp_dy_param_double;
    }
    if (dy_param_string_ != tmp_dy_param_string) {
        cout << "configuration changed: dy_param_string [ " << dy_param_string_ << " -> " << tmp_dy_param_string
             << " ]." << endl;
        dy_param_string_ = tmp_dy_param_string;
    }
    if (dy_run_number_ != tmp_dy_run_number) {
        cout << "configuration changed: dy_run_number [ " << dy_run_number_ << " -> " << tmp_dy_run_number << " ]."
             << endl;
        dy_run_number_ = tmp_dy_run_number;
    }

    config_mtime_ = conf_mtime;

    lock_guard<mutex> dynamic_config_json_lg(dynamic_config_json_mtx_);
    dynamic_config_json_["dy_run_number"] = json::value::number(dy_run_number_);
    dynamic_config_json_["dy_param_int"] = json::value::number(dy_param_int_);
    dynamic_config_json_["dy_param_double"] = json::value::number(dy_param_double_);
    dynamic_config_json_["dy_param_string"] = json::value::string(dy_param_string_);
    dynamic_config_json_["config_mtime"] = json::value::string(boost::trim_copy(string(ctime(&config_mtime_))));

    return true;
}

int diffraflow::IngConfig::get_dy_run_number() { return dy_run_number_.load(); }

int diffraflow::IngConfig::get_dy_param_int() { return dy_param_int_.load(); }

double diffraflow::IngConfig::get_dy_param_double() { return dy_param_double_.load(); }

string diffraflow::IngConfig::get_dy_param_string() {
    lock_guard<mutex> lg(dy_param_string_mtx_);
    return dy_param_string_;
}

bool diffraflow::IngConfig::metrics_pulsar_params_are_set() {
    return (!metrics_pulsar_broker_address.empty() && !metrics_pulsar_topic_name.empty() &&
            !metrics_pulsar_message_key.empty());
}

bool diffraflow::IngConfig::metrics_http_params_are_set() {
    return (!metrics_http_host.empty() && metrics_http_port > 0);
}
