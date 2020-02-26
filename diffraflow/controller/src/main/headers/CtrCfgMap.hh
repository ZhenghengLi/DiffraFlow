#ifndef __CtrCfgMap_H__
#define __CtrCfgMap_H__

#include <map>
#include <string>
#include <log4cxx/logger.h>
#include <ctime>

#include "GenericConfiguration.hh"

using std::map;
using std::string;

namespace diffraflow {
    class CtrCfgMap: public GenericConfiguration {
    public:
        CtrCfgMap();
        ~CtrCfgMap();

        bool load(const char* filename) override;
        void print() override;

    public:
        map<string, string> data;
        time_t              mtime;

    private:
        static log4cxx::LoggerPtr logger_;
    };
}

#endif