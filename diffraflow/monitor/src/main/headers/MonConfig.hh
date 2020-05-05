#ifndef __MonConfig__
#define __MonConfig__

#include "DynamicConfiguration.hh"
#include <log4cxx/logger.h>
#include <string>
#include <atomic>
#include <mutex>

using std::atomic;
using std::mutex;

namespace diffraflow {
    class MonConfig : public DynamicConfiguration {
    public:
        MonConfig();
        ~MonConfig();

        bool load(const char* filename) override;
        void print() override;

        bool zookeeper_setting_is_ready();

        bool metrics_pulsar_params_are_set();
        bool metrics_http_params_are_set();

        json::value collect_metrics() override;

    public:
        int get_dy_param_int();
        double get_dy_param_double();
        string get_dy_param_string();

    protected:
        bool check_and_commit_(const map<string, string>& conf_map, const time_t conf_mtime) override;

    public:
        // static parameters
        string node_name;
        int monitor_id;
        string image_http_host;
        int image_http_port;
        int request_timeout;

        string metrics_pulsar_broker_address;
        string metrics_pulsar_topic_name;
        string metrics_pulsar_message_key;
        int metrics_pulsar_report_period;
        string metrics_http_host;
        int metrics_http_port;

    private:
        // dynamic parameters
        atomic<int> dy_param_int_;
        atomic<double> dy_param_double_;

        string dy_param_string_;
        mutex dy_param_string_mtx_;

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