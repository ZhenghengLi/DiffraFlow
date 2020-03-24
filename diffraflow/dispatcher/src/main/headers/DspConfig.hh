#ifndef DspConfig_H
#define DspConfig_H

#include "GenericConfiguration.hh"
#include <log4cxx/logger.h>
#include <string>

#include "DspSender.hh"
#include "MetricsProvider.hh"

using std::string;

namespace diffraflow {
    class DspConfig: public GenericConfiguration, public MetricsProvider {
    public:
        DspConfig();
        ~DspConfig();

        bool load(const char* filename) override;
        void print() override;

    public:
        bool pulsar_params_are_set();
        bool http_server_params_are_set();

    public:
        json::value collect_metrics() override;

    public:
        uint32_t                    dispatcher_id;
        string                      listen_host;
        int                         listen_port;
        DspSender::CompressMethod   compress_method;
        int                         compress_level;
        string                      pulsar_broker_address;
        string                      pulsar_topic_name;
        string                      pulsar_message_key;
        size_t                      pulsar_report_period;
        string                      http_server_host;
        int                         http_server_port;

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif