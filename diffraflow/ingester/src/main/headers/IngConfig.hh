#ifndef __IngConfig__
#define __IngConfig__

#include "DynamicConfiguration.hh"
#include <log4cxx/logger.h>
#include <string>
#include <atomic>
#include <mutex>

using std::atomic;
using std::mutex;

namespace diffraflow {
    class IngConfig: public DynamicConfiguration {
    public:
        IngConfig();
        ~IngConfig();

        bool load(const char* filename) override;
        void print() override;

        bool zookeeper_setting_is_ready();

        json::value collect_metrics() override;

    public:
        int    get_dy_param_int();
        double get_dy_param_double();
        string get_dy_param_string();

    protected:
        bool check_and_commit_(const map<string, string>& conf_map,
            const time_t conf_mtime) override;

    public:
        // static parameters
        uint32_t ingester_id;
        string   combiner_host;
        int      combiner_port;

    private:
        // dynamic parameters
        atomic<int>     dy_param_int_;
        atomic<double>  dy_param_double_;

        string  dy_param_string_;
        mutex   dy_param_string_mtx_;

        time_t config_mtime_;

        bool zookeeper_setting_ready_flag_;

        json::value ingester_config_json_;
        mutex       ingester_config_json_mtx_;

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif