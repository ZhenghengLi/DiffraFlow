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
        uint32_t sender_id;
        string listen_host;
        int listen_port;

        string data_dir;
        int max_seq_num;
        int events_per_file;
        int total_events;

        string dispatcher_host; // can be also configured by nodemap but with lower priority
        int dispatcher_port;    // can be also configured by nodemap but with lower priority
        int module_id;          // can be also configured by nodemap but with lower priority

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif
