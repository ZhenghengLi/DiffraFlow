#ifndef __AggConfig_H__
#define __AggConfig_H__

#include "GenericConfiguration.hh"
#include <log4cxx/logger.h>
#include <string>

#include "MetricsProvider.hh"

using std::string;

namespace diffraflow {
    class AggConfig : public GenericConfiguration, public MetricsProvider {
    public:
        AggConfig();
        ~AggConfig();

        bool load(const char* filename) override;
        void print() override;

    public:
        bool metrics_pulsar_params_are_set();
        bool metrics_http_params_are_set();

    public:
        json::value collect_metrics() override;

    public:
        string http_server_host;
        int http_server_port;

        string pulsar_url;
        string subscription_name;
        string controller_topic;
        string sender_topic;
        string dispatcher_topic;
        string combiner_topic;
        string ingester_topic;
        string monitor_topic;

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