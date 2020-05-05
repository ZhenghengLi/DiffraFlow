#ifndef __CtrConfig_H__
#define __CtrConfig_H__

#include <string>
#include <log4cxx/logger.h>

#include "GenericConfiguration.hh"

using std::map;
using std::string;

namespace diffraflow {
    class CtrConfig : public GenericConfiguration {
    public:
        CtrConfig();
        ~CtrConfig();

        bool load(const char* filename) override;
        void print() override;

    public:
        string http_host;
        int http_port;
        int request_timeout;

    private:
        static log4cxx::LoggerPtr logger_;
    };
} // namespace diffraflow

#endif