#ifndef CmbConfig_H
#define CmbConfig_H

#include "GenericConfiguration.hh"
#include "MetricsProvider.hh"
#include <log4cxx/logger.h>

#include <map>

using std::map;
using std::string;

namespace diffraflow {
    class CmbConfig: public GenericConfiguration, public MetricsProvider {
    public:
       CmbConfig();
       ~CmbConfig();
       bool load(const char* filename) override;
       void print() override;

    public:
        bool metrics_pulsar_params_are_set();
        bool metrics_http_params_are_set();

    public:
        json::value collect_metrics() override;

    public:
        string imgfrm_listen_host;
        int    imgfrm_listen_port;

        string imgdat_listen_host;
        int    imgdat_listen_port;

        size_t imgdat_queue_capacity;

        string metrics_pulsar_broker_address;
        string metrics_pulsar_topic_name;
        string metrics_pulsar_message_key;
        int    metrics_pulsar_report_period;
        string metrics_http_host;
        int    metrics_http_port;

    private:
        json::value static_config_json_;
        json::value metrics_config_json_;

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif
