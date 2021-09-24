#ifndef __IngConfig__
#define __IngConfig__

#include "DynamicConfiguration.hh"
#include <log4cxx/logger.h>
#include <string>
#include <atomic>
#include <mutex>

#include "FeatureExtraction.hh"

using std::atomic;
using std::mutex;

namespace diffraflow {
    class IngConfig : public DynamicConfiguration {
    public:
        IngConfig();
        ~IngConfig();

        bool load(const char* filename) override;
        void print() override;

        bool zookeeper_setting_is_ready();

        bool metrics_pulsar_params_are_set();
        bool metrics_http_params_are_set();

        json::value collect_metrics() override;

    public:
        int get_dy_run_number();

        int get_dy_param_int();
        double get_dy_param_double();
        string get_dy_param_string();

        FeatureExtraction::PeakMsseParams get_dy_peak_msse_params();
        float get_dy_mean_rms_min_energy();
        float get_dy_mean_rms_max_energy();

    protected:
        bool check_and_commit_(const map<string, string>& conf_map, const time_t conf_mtime) override;

    public:
        // static parameters
        string storage_dir;
        bool save_calib_data;
        bool save_raw_data;
        string node_name;
        int ingester_id;
        int hdf5_chunk_size;
        int hdf5_compress_level;
        bool hdf5_swmr_mode;
        int file_imgcnt_limit;
        int file_imgcnt_rand;

        string combiner_host;
        int combiner_port;
        string combiner_sock;

        string image_http_host;
        int image_http_port;
        size_t recnxn_wait_time;
        size_t recnxn_max_count;

        size_t buffer_capacity;
        size_t queue_capacity_raw;
        size_t queue_capacity_calib;
        size_t queue_capacity_feature;
        size_t queue_capacity_write;

        string calib_param_file;
        bool gpu_enable;
        int gpu_device_index;

        string metrics_pulsar_broker_address;
        string metrics_pulsar_topic_name;
        string metrics_pulsar_message_key;
        int metrics_pulsar_report_period;
        string metrics_http_host;
        int metrics_http_port;

    private:
        // dynamic parameters
        atomic<int> dy_run_number_;

        atomic<int> dy_param_int_;
        atomic<double> dy_param_double_;

        string dy_param_string_;
        mutex dy_param_string_mtx_;

        // peak msse parameters
        float dy_peak_msse_min_energy_;
        float dy_peak_msse_max_energy_;
        float dy_peak_msse_inlier_thr_;
        float dy_peak_msse_outlier_thr_;
        float dy_peak_msse_residual_thr_;
        float dy_peak_msse_energy_thr_;
        mutex dy_peak_msse_params_mtx_;

        // mean rms parameters
        atomic<float> dy_mean_rms_min_energy_;
        atomic<float> dy_mean_rms_max_energy_;

        time_t config_mtime_;

        bool zookeeper_setting_ready_flag_;

        json::value static_config_json_;
        json::value metrics_config_json_;
        json::value dynamic_config_json_;
        mutex dynamic_config_json_mtx_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif