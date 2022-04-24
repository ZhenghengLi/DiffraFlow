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

    buffer_capacity = 500;
    queue_capacity_raw = 30;
    queue_capacity_calib = 30;
    queue_capacity_feature = 30;
    queue_capacity_write = 400;

    save_calib_data = false;
    save_raw_data = false;

    gpu_enable = false;
    gpu_device_index = 0;

    hdf5_chunk_size = 1;
    hdf5_compress_level = 0;
    hdf5_swmr_mode = true;
    file_imgcnt_limit = 1000;
    file_imgcnt_rand = 0;

    zookeeper_setting_ready_flag_ = false;

    metrics_pulsar_report_period = 1000;
    metrics_http_port = -1;

    // initial values of dynamic configurations
    dy_run_number_ = 0;

    dy_peak_msse_min_energy_ = -10;
    dy_peak_msse_max_energy_ = 1000;
    dy_peak_msse_inlier_thr_ = 2;
    dy_peak_msse_outlier_thr_ = 10;
    dy_peak_msse_residual_thr_ = 50;
    dy_peak_msse_energy_thr_ = 0;

    dy_mean_rms_min_energy_ = -10;
    dy_mean_rms_max_energy_ = 1000;

    dy_saving_global_mean_thr_ = 0;
    dy_saving_global_rms_thr_ = 0;
    dy_saving_peak_pixels_thr_ = 100;

    dy_monitor_global_mean_thr_ = 0;
    dy_monitor_global_rms_thr_ = 0;
    dy_monitor_peak_pixels_thr_ = 100;
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
        LOG4CXX_WARN(logger_, "zookeeper setting is not ready, configuration will not be dynamically updated.");
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
        } else if (key == "buffer_capacity") {
            buffer_capacity = atoi(value.c_str());
        } else if (key == "queue_capacity_raw") {
            queue_capacity_raw = atoi(value.c_str());
        } else if (key == "queue_capacity_calib") {
            queue_capacity_calib = atoi(value.c_str());
        } else if (key == "queue_capacity_feature") {
            queue_capacity_feature = atoi(value.c_str());
        } else if (key == "queue_capacity_write") {
            queue_capacity_write = atoi(value.c_str());
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
    if (buffer_capacity < 1 || buffer_capacity > 1000) {
        LOG4CXX_ERROR(logger_, "buffer_capacity is out of range " << 1 << "-" << 1000);
        succ_flag = false;
    }
    if (queue_capacity_raw < 1 || queue_capacity_raw > 1000) {
        LOG4CXX_ERROR(logger_, "queue_capacity_raw is out of range " << 1 << "-" << 1000);
        succ_flag = false;
    }
    if (queue_capacity_calib < 1 || queue_capacity_calib > 1000) {
        LOG4CXX_ERROR(logger_, "queue_capacity_calib is out of range " << 1 << "-" << 1000);
        succ_flag = false;
    }
    if (queue_capacity_feature < 1 || queue_capacity_feature > 1000) {
        LOG4CXX_ERROR(logger_, "queue_capacity_feature is out of range " << 1 << "-" << 1000);
        succ_flag = false;
    }
    if (queue_capacity_write < 1 || queue_capacity_write > 1000) {
        LOG4CXX_ERROR(logger_, "queue_capacity_write is out of range " << 1 << "-" << 1000);
        succ_flag = false;
    }
    if (gpu_enable) {
        if (gpu_device_index < 0) {
            LOG4CXX_ERROR(logger_, "gpu_device_index < 0 while gpu_enable = true");
            succ_flag = false;
        } else {
            int device_count = 0;
            cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
            if (cuda_err == cudaSuccess) {
                if (gpu_device_index >= device_count) {
                    LOG4CXX_ERROR(logger_, "gpu_device_index >= " << device_count << " while gpu_enable = true");
                    succ_flag = false;
                }
            } else {
                LOG4CXX_ERROR(logger_, "Failed to get gpu device count with error: " << cudaGetErrorString(cuda_err));
                succ_flag = false;
            }
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
        static_config_json_["buffer_capacity"] = json::value::number((uint32_t)buffer_capacity);
        static_config_json_["queue_capacity_raw"] = json::value::number((uint32_t)queue_capacity_raw);
        static_config_json_["queue_capacity_calib"] = json::value::number((uint32_t)queue_capacity_calib);
        static_config_json_["queue_capacity_feature"] = json::value::number((uint32_t)queue_capacity_feature);
        static_config_json_["queue_capacity_write"] = json::value::number((uint32_t)queue_capacity_write);
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

bool diffraflow::IngConfig::metrics_pulsar_params_are_set() {
    return (!metrics_pulsar_broker_address.empty() && !metrics_pulsar_topic_name.empty() &&
            !metrics_pulsar_message_key.empty());
}

bool diffraflow::IngConfig::metrics_http_params_are_set() {
    return (!metrics_http_host.empty() && metrics_http_port > 0);
}

void diffraflow::IngConfig::print() {
    // with all locks
    lock_guard<mutex> dy_peak_msse_params_lg(dy_peak_msse_params_mtx_);

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
    cout << "- buffer_capacity = " << buffer_capacity << endl;
    cout << "- queue_capacity_raw = " << queue_capacity_raw << endl;
    cout << "- queue_capacity_calib = " << queue_capacity_calib << endl;
    cout << "- queue_capacity_feature = " << queue_capacity_feature << endl;
    cout << "- queue_capacity_write = " << queue_capacity_write << endl;
    cout << "- calib_param_file = " << calib_param_file << endl;
    cout << "- gpu_enable = " << gpu_enable << endl;
    cout << "- gpu_device_index = " << gpu_device_index << endl;
    cout << "- storage_dir = " << storage_dir << endl;
    cout << "- save_calib_data = " << save_calib_data << endl;
    cout << "- save_raw_data = " << save_raw_data << endl;
    cout << "dynamic parameters:" << endl;
    cout << "- dy_run_number = " << dy_run_number_.load() << endl;
    cout << "- dy_peak_msse_min_energy = " << dy_peak_msse_min_energy_ << endl;
    cout << "- dy_peak_msse_max_energy = " << dy_peak_msse_max_energy_ << endl;
    cout << "- dy_peak_msse_inlier_thr = " << dy_peak_msse_inlier_thr_ << endl;
    cout << "- dy_peak_msse_outlier_thr = " << dy_peak_msse_outlier_thr_ << endl;
    cout << "- dy_peak_msse_residual_thr = " << dy_peak_msse_residual_thr_ << endl;
    cout << "- dy_peak_msse_energy_thr = " << dy_peak_msse_energy_thr_ << endl;
    cout << "- dy_mean_rms_min_energy = " << dy_mean_rms_min_energy_ << endl;
    cout << "- dy_mean_rms_max_energy = " << dy_mean_rms_max_energy_ << endl;
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
    lock_guard<mutex> dy_peak_msse_params_lg(dy_peak_msse_params_mtx_);

    // values before commit
    int tmp_dy_run_number = dy_run_number_.load();

    float tmp_dy_peak_msse_min_energy = dy_peak_msse_min_energy_;
    float tmp_dy_peak_msse_max_energy = dy_peak_msse_max_energy_;
    float tmp_dy_peak_msse_inlier_thr = dy_peak_msse_inlier_thr_;
    float tmp_dy_peak_msse_outlier_thr = dy_peak_msse_outlier_thr_;
    float tmp_dy_peak_msse_residual_thr = dy_peak_msse_residual_thr_;
    float tmp_dy_peak_msse_energy_thr = dy_peak_msse_energy_thr_;

    float tmp_dy_mean_rms_min_energy = dy_mean_rms_min_energy_.load();
    float tmp_dy_mean_rms_max_energy = dy_mean_rms_max_energy_.load();

    float tmp_dy_saving_global_mean_thr = dy_saving_global_mean_thr_.load();
    float tmp_dy_saving_global_rms_thr = dy_saving_global_rms_thr_.load();
    int tmp_dy_saving_peak_pixels_thr = dy_saving_peak_pixels_thr_.load();

    float tmp_dy_monitor_global_mean_thr = dy_monitor_global_mean_thr_.load();
    float tmp_dy_monitor_global_rms_thr = dy_monitor_global_rms_thr_.load();
    int tmp_dy_monitor_peak_pixels_thr = dy_monitor_peak_pixels_thr_.load();

    // convert
    for (map<string, string>::const_iterator iter = conf_map.begin(); iter != conf_map.end(); ++iter) {
        string key = iter->first;
        string value = iter->second;
        // run number
        if (key == "dy_run_number") {
            tmp_dy_run_number = atoi(value.c_str());
            // peak msse
        } else if (key == "dy_peak_msse_min_energy") {
            tmp_dy_peak_msse_min_energy = atof(value.c_str());
        } else if (key == "dy_peak_msse_max_energy") {
            tmp_dy_peak_msse_max_energy = atof(value.c_str());
        } else if (key == "dy_peak_msse_inlier_thr") {
            tmp_dy_peak_msse_inlier_thr = atof(value.c_str());
        } else if (key == "dy_peak_msse_outlier_thr") {
            tmp_dy_peak_msse_outlier_thr = atof(value.c_str());
        } else if (key == "dy_peak_msse_residual_thr") {
            tmp_dy_peak_msse_residual_thr = atof(value.c_str());
        } else if (key == "dy_peak_msse_energy_thr") {
            tmp_dy_peak_msse_energy_thr = atof(value.c_str());
            // global mean and rms
        } else if (key == "dy_mean_rms_min_energy") {
            tmp_dy_mean_rms_min_energy = atof(value.c_str());
        } else if (key == "dy_mean_rms_max_energy") {
            tmp_dy_mean_rms_max_energy = atof(value.c_str());
            // saving filter thresholds
        } else if (key == "dy_saving_global_mean_thr") {
            tmp_dy_saving_global_mean_thr = atof(value.c_str());
        } else if (key == "dy_saving_global_rms_thr") {
            tmp_dy_saving_global_rms_thr = atof(value.c_str());
        } else if (key == "dy_saving_peak_pixels_thr") {
            tmp_dy_saving_peak_pixels_thr = atoi(value.c_str());
            // monitor filter thresholds
        } else if (key == "dy_monitor_global_mean_thr") {
            tmp_dy_monitor_global_mean_thr = atof(value.c_str());
        } else if (key == "dy_monitor_global_rms_thr") {
            tmp_dy_monitor_global_rms_thr = atof(value.c_str());
        } else if (key == "dy_monitor_peak_pixels_thr") {
            tmp_dy_monitor_peak_pixels_thr = atoi(value.c_str());
        }
    }

    // validation check
    bool invalid_flag = false;
    if (tmp_dy_run_number < 0) {
        // cppcheck-suppress shiftNegative
        LOG4CXX_WARN(logger_, "invalid configuration: dy_run_number(" << tmp_dy_run_number << ") is less than zero.");
        invalid_flag = true;
    }
    if (tmp_dy_peak_msse_min_energy >= tmp_dy_peak_msse_max_energy) {
        LOG4CXX_WARN(logger_, "invalid configuration: dy_peak_msse_min_energy(" << tmp_dy_peak_msse_min_energy
                                                                                << ") >= dy_peak_msse_max_energy("
                                                                                << tmp_dy_peak_msse_max_energy << ").");
        invalid_flag = true;
    }
    if (tmp_dy_peak_msse_inlier_thr >= tmp_dy_peak_msse_outlier_thr) {
        LOG4CXX_WARN(logger_, "invalid configuration: dy_peak_msse_inlier_thr("
                                  << tmp_dy_peak_msse_inlier_thr << ") >= dy_peak_msse_outlier_thr("
                                  << tmp_dy_peak_msse_outlier_thr << ").");
        invalid_flag = true;
    }
    if (tmp_dy_peak_msse_inlier_thr <= 0) {
        LOG4CXX_WARN(
            logger_, "invalid configuration: dy_peak_msse_inlier_thr(" << tmp_dy_peak_msse_inlier_thr << ") <= 0.");
        invalid_flag = true;
    }
    if (tmp_dy_peak_msse_outlier_thr <= 0) {
        LOG4CXX_WARN(
            logger_, "invalid configuration: dy_peak_msse_outlier_thr(" << tmp_dy_peak_msse_outlier_thr << ") <= 0.");
        invalid_flag = true;
    }
    if (tmp_dy_mean_rms_min_energy >= tmp_dy_mean_rms_max_energy) {
        LOG4CXX_WARN(logger_, "invalid configuration: dy_mean_rms_min_energy(" << tmp_dy_mean_rms_min_energy
                                                                               << ") >= dy_mean_rms_max_energy("
                                                                               << tmp_dy_mean_rms_max_energy << ").");
        invalid_flag = true;
    }

    if (invalid_flag) {
        return false;
    }

    // commit change
    // - run number
    if (dy_run_number_ != tmp_dy_run_number) {
        LOG4CXX_WARN(logger_,
            "configuration changed: dy_run_number [ " << dy_run_number_ << " -> " << tmp_dy_run_number << " ].");
        dy_run_number_ = tmp_dy_run_number;
    }
    // - peak msse
    if (dy_peak_msse_min_energy_ != tmp_dy_peak_msse_min_energy) {
        LOG4CXX_WARN(logger_, "configuration changed: dy_peak_msse_min_energy [ "
                                  << dy_peak_msse_min_energy_ << " -> " << tmp_dy_peak_msse_min_energy << " ].");
        dy_peak_msse_min_energy_ = tmp_dy_peak_msse_min_energy;
    }
    if (dy_peak_msse_max_energy_ != tmp_dy_peak_msse_max_energy) {
        LOG4CXX_WARN(logger_, "configuration changed: dy_peak_msse_max_energy [ "
                                  << dy_peak_msse_max_energy_ << " -> " << tmp_dy_peak_msse_max_energy << " ].");
        dy_peak_msse_max_energy_ = tmp_dy_peak_msse_max_energy;
    }
    if (dy_peak_msse_inlier_thr_ != tmp_dy_peak_msse_inlier_thr) {
        LOG4CXX_WARN(logger_, "configuration changed: dy_peak_msse_inlier_thr ["
                                  << dy_peak_msse_inlier_thr_ << " -> " << tmp_dy_peak_msse_inlier_thr << " ].");
        dy_peak_msse_inlier_thr_ = tmp_dy_peak_msse_inlier_thr;
    }
    if (dy_peak_msse_outlier_thr_ != tmp_dy_peak_msse_outlier_thr) {
        LOG4CXX_WARN(logger_, "configuration changed: dy_peak_msse_outlier_thr ["
                                  << dy_peak_msse_outlier_thr_ << " -> " << tmp_dy_peak_msse_outlier_thr << " ].");
        dy_peak_msse_outlier_thr_ = tmp_dy_peak_msse_outlier_thr;
    }
    if (dy_peak_msse_residual_thr_ != tmp_dy_peak_msse_residual_thr) {
        LOG4CXX_WARN(logger_, "configuration changed: dy_peak_msse_residual_thr ["
                                  << dy_peak_msse_residual_thr_ << " -> " << tmp_dy_peak_msse_residual_thr << " ].");
        dy_peak_msse_residual_thr_ = tmp_dy_peak_msse_residual_thr;
    }
    if (dy_peak_msse_energy_thr_ != tmp_dy_peak_msse_energy_thr) {
        LOG4CXX_WARN(logger_, "configuration changed: dy_peak_msse_energy_thr ["
                                  << dy_peak_msse_energy_thr_ << " -> " << tmp_dy_peak_msse_energy_thr << " ].");
        dy_peak_msse_energy_thr_ = tmp_dy_peak_msse_energy_thr;
    }
    // - global mean and rms
    if (dy_mean_rms_min_energy_ != tmp_dy_mean_rms_min_energy) {
        LOG4CXX_WARN(logger_, "configuration changed: dy_mean_rms_min_energy [" << dy_mean_rms_min_energy_ << " -> "
                                                                                << tmp_dy_mean_rms_min_energy << " ].");
        dy_mean_rms_min_energy_ = tmp_dy_mean_rms_min_energy;
    }
    if (dy_mean_rms_max_energy_ != tmp_dy_mean_rms_max_energy) {
        LOG4CXX_WARN(logger_, "configuration changed: dy_mean_rms_max_energy [" << dy_mean_rms_max_energy_ << " -> "
                                                                                << tmp_dy_mean_rms_max_energy << " ].");
        dy_mean_rms_max_energy_ = tmp_dy_mean_rms_max_energy;
    }
    // - saving filter thresholds
    if (dy_saving_global_mean_thr_ != tmp_dy_saving_global_mean_thr) {
        LOG4CXX_WARN(logger_, "configuration changed: dy_saving_global_mean_thr ["
                                  << dy_saving_global_mean_thr_ << " -> " << tmp_dy_saving_global_mean_thr << " ].");
        dy_saving_global_mean_thr_ = tmp_dy_saving_global_mean_thr;
    }
    if (dy_saving_global_rms_thr_ != tmp_dy_saving_global_rms_thr) {
        LOG4CXX_WARN(logger_, "configuration changed: dy_saving_global_rms_thr ["
                                  << dy_saving_global_rms_thr_ << " -> " << tmp_dy_saving_global_rms_thr << " ].");
        dy_saving_global_rms_thr_ = tmp_dy_saving_global_rms_thr;
    }
    if (dy_saving_peak_pixels_thr_ != tmp_dy_saving_peak_pixels_thr) {
        LOG4CXX_WARN(logger_, "configuration changed: dy_saving_peak_pixels_thr ["
                                  << dy_saving_peak_pixels_thr_ << " -> " << tmp_dy_saving_peak_pixels_thr << " ].");
        dy_saving_peak_pixels_thr_ = tmp_dy_saving_peak_pixels_thr;
    }
    // - monitor filter thresholds
    if (dy_monitor_global_mean_thr_ != tmp_dy_monitor_global_mean_thr) {
        LOG4CXX_WARN(logger_, "configuration changed: dy_monitor_global_mean_thr ["
                                  << dy_monitor_global_mean_thr_ << " -> " << tmp_dy_monitor_global_mean_thr << " ].");
        dy_monitor_global_mean_thr_ = tmp_dy_monitor_global_mean_thr;
    }
    if (dy_monitor_global_rms_thr_ != tmp_dy_monitor_global_rms_thr) {
        LOG4CXX_WARN(logger_, "configuration changed: dy_monitor_global_rms_thr ["
                                  << dy_monitor_global_rms_thr_ << " -> " << tmp_dy_monitor_global_rms_thr << " ].");
        dy_monitor_global_rms_thr_ = tmp_dy_monitor_global_rms_thr;
    }
    if (dy_monitor_peak_pixels_thr_ != tmp_dy_monitor_peak_pixels_thr) {
        LOG4CXX_WARN(logger_, "configuration changed: dy_monitor_peak_pixels_thr ["
                                  << dy_monitor_peak_pixels_thr_ << " -> " << tmp_dy_monitor_peak_pixels_thr << " ].");
        dy_monitor_peak_pixels_thr_ = tmp_dy_monitor_peak_pixels_thr;
    }

    config_mtime_ = conf_mtime;

    lock_guard<mutex> dynamic_config_json_lg(dynamic_config_json_mtx_);
    // run number
    dynamic_config_json_["dy_run_number"] = json::value::number(dy_run_number_.load());
    // peak msse
    dynamic_config_json_["dy_peak_msse_min_energy"] = json::value::number(dy_peak_msse_min_energy_);
    dynamic_config_json_["dy_peak_msse_max_energy"] = json::value::number(dy_peak_msse_max_energy_);
    dynamic_config_json_["dy_peak_msse_inlier_thr"] = json::value::number(dy_peak_msse_inlier_thr_);
    dynamic_config_json_["dy_peak_msse_outlier_thr"] = json::value::number(dy_peak_msse_outlier_thr_);
    dynamic_config_json_["dy_peak_msse_residual_thr"] = json::value::number(dy_peak_msse_residual_thr_);
    dynamic_config_json_["dy_peak_msse_energy_thr"] = json::value::number(dy_peak_msse_energy_thr_);
    // global mean and rms
    dynamic_config_json_["dy_mean_rms_min_energy"] = json::value::number(dy_mean_rms_min_energy_.load());
    dynamic_config_json_["dy_mean_rms_max_energy"] = json::value::number(dy_mean_rms_max_energy_.load());
    // saving filter thresholds
    dynamic_config_json_["dy_saving_global_mean_thr"] = json::value::number(dy_saving_global_mean_thr_.load());
    dynamic_config_json_["dy_saving_global_rms_thr"] = json::value::number(dy_saving_global_rms_thr_.load());
    dynamic_config_json_["dy_saving_peak_pixels_thr"] = json::value::number(dy_saving_peak_pixels_thr_.load());
    // monitor filter thresholds
    dynamic_config_json_["dy_monitor_global_mean_thr"] = json::value::number(dy_monitor_global_mean_thr_.load());
    dynamic_config_json_["dy_monitor_global_rms_thr"] = json::value::number(dy_monitor_global_rms_thr_.load());
    dynamic_config_json_["dy_monitor_peak_pixels_thr"] = json::value::number(dy_monitor_peak_pixels_thr_.load());
    // mtime
    dynamic_config_json_["config_mtime"] = json::value::string(boost::trim_copy(string(ctime(&config_mtime_))));

    return true;
}

int diffraflow::IngConfig::get_dy_run_number() { return dy_run_number_.load(); }

diffraflow::FeatureExtraction::PeakMsseParams diffraflow::IngConfig::get_dy_peak_msse_params() {
    lock_guard<mutex> lg(dy_peak_msse_params_mtx_);
    return FeatureExtraction::PeakMsseParams{dy_peak_msse_min_energy_, dy_peak_msse_max_energy_,
        dy_peak_msse_inlier_thr_, dy_peak_msse_outlier_thr_, dy_peak_msse_residual_thr_, dy_peak_msse_energy_thr_};
}

float diffraflow::IngConfig::get_dy_mean_rms_min_energy() { return dy_mean_rms_min_energy_.load(); }

float diffraflow::IngConfig::get_dy_mean_rms_max_energy() { return dy_mean_rms_max_energy_.load(); }

float diffraflow::IngConfig::get_dy_saving_global_mean_thr() { return dy_saving_global_mean_thr_.load(); }

float diffraflow::IngConfig::get_dy_saving_global_rms_thr() { return dy_saving_global_rms_thr_.load(); }

int diffraflow::IngConfig::get_dy_saving_peak_pixels_thr() { return dy_saving_peak_pixels_thr_.load(); }

float diffraflow::IngConfig::get_dy_monitor_global_mean_thr() { return dy_monitor_global_mean_thr_.load(); }

float diffraflow::IngConfig::get_dy_monitor_global_rms_thr() { return dy_monitor_global_rms_thr_.load(); }

int diffraflow::IngConfig::get_dy_monitor_peak_pixels_thr() { return dy_monitor_peak_pixels_thr_.load(); }
