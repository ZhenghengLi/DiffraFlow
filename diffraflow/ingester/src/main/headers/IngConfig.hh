#ifndef __IngConfig__
#define __IngConfig__

#include "DynamicConfiguration.hh"
#include <log4cxx/logger.h>
#include <string>

namespace diffraflow {
    class IngConfig: public DynamicConfiguration {
    public:
        IngConfig();
        ~IngConfig();

        bool load(const char* filename) override;
        void print() override;

        void convert_and_check() override;

    public:
        int     ingester_id;
        string  listen_host;
        int     listen_port;

        map<string, string> conf_map_in_use;

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif