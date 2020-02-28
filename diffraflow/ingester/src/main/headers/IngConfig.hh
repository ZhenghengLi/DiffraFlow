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

    protected:
        void convert_and_check_() override;

    public:
        map<string, string> conf_map_in_use;

        // static parameters
        int     ingester_id;
        string  combiner_host;
        int     combiner_port;

        // dynamic parameters
        int     dy_param_int;
        double  dy_param_double;
        string  dy_param_string;

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif