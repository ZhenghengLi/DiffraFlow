#ifndef SndConfig_H
#define SndConfig_H

#include "GenericConfiguration.hh"
#include <log4cxx/logger.h>
#include <string>
#include <map>

using std::string;
using std::map;

namespace diffraflow {
    class SndConfig : public GenericConfiguration {
    public:
        SndConfig();
        ~SndConfig();

        bool load(const char* filename) override;
        void print() override;

        bool load_nodemap(const char* filename, const string nodename);

    public:
        bool metrics_pulsar_params_are_set();
        bool metrics_http_params_are_set();

    public:
        uint32_t sender_id;
        string listen_host;
        int listen_port;

        string data_dir;
        int events_per_file;
        int total_events;

        string dispatcher_host; // can be also configured by nodemap but with lower priority
        int dispatcher_port;    // can be also configured by nodemap but with lower priority
        int module_id;          // can be also configured by nodemap but with lower priority

        string metrics_pulsar_broker_address;
        string metrics_pulsar_topic_name;
        string metrics_pulsar_message_key;
        int metrics_pulsar_report_period;
        string metrics_http_host;
        int metrics_http_port;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
