#ifndef DspConfig_H
#define DspConfig_H

#include "GenericConfiguration.hh"
#include <log4cxx/logger.h>
#include <string>

#include "MetricsProvider.hh"

using std::string;

namespace diffraflow {
    class DspConfig : public GenericConfiguration, public MetricsProvider {
    public:
        DspConfig();
        ~DspConfig();

        bool load(const char* filename) override;
        void print() override;

    public:
        bool metrics_pulsar_params_are_set();
        bool metrics_http_params_are_set();

    public:
        json::value collect_metrics() override;

    public:
        uint32_t dispatcher_id;
        string listen_host;
        int listen_port;
        int max_queue_size;

        string metrics_pulsar_broker_address;
        string metrics_pulsar_topic_name;
        string metrics_pulsar_message_key;
        int metrics_pulsar_report_period;
        string metrics_http_host;
        int metrics_http_port;

    private:
        json::value static_config_json_;
        json::value metrics_config_json_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
