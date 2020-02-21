#ifndef __DynamicConfiguration_H__
#define __DynamicConfiguration_H__

#include "GenericConfiguration.hh"
#include <map>
#include <string>
#include <log4cxx/logger.h>

using std::map;
using std::string;

namespace diffraflow {
    class DynamicConfiguration: public GenericConfiguration {
    public:
        DynamicConfiguration();
        ~DynamicConfiguration();

        virtual bool load(const char* filename);
        virtual void print();

    protected:
        string zookeeper_server_;
        int zookeeper_expiration_time_;
        map<string, string> conf_map_;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif