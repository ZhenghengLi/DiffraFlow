#ifndef DspConfig_H
#define DspConfig_H

#include "GenericConfiguration.hh"
#include <log4cxx/logger.h>
#include <string>

#include "DspSender.hh"

using std::string;

namespace diffraflow {
    class DspConfig: public GenericConfiguration {
    public:
        DspConfig();
        ~DspConfig();

        bool load(const char* filename);
        void print();

    public:
        uint32_t                    dispatcher_id;
        string                      listen_host;
        int                         listen_port;
        DspSender::CompressMethod   compress_method;
        int                         compress_level;

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif