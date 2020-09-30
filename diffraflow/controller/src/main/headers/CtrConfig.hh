#ifndef __CtrConfig_H__
#define __CtrConfig_H__

#include <string>
#include <log4cxx/logger.h>

#include "GenericConfiguration.hh"
#include "MetricsProvider.hh"

using std::map;
using std::string;

namespace diffraflow {
    class CtrConfig : public GenericConfiguration, public MetricsProvider {
    public:
        CtrConfig();
        ~CtrConfig();

        bool load(const char* filename) override;
        void print() override;

    public:
        bool metrics_pulsar_params_are_set();
        bool metrics_http_params_are_set();

    public:
        json::value collect_metrics() override;

    public:
        string http_host;
        int http_port;
        int request_timeout;

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