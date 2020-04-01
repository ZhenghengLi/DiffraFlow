#ifndef CmbConfig_H
#define CmbConfig_H

#include "GenericConfiguration.hh"
#include <log4cxx/logger.h>

#include <map>

using std::map;
using std::string;

namespace diffraflow {
    class CmbConfig: public GenericConfiguration {
    public:
       CmbConfig();
       ~CmbConfig();
       bool load(const char* filename);
       void print();

    public:
        string imgfrm_listen_host;
        int    imgfrm_listen_port;

        string imgdat_listen_host;
        int    imgdat_listen_port;

    private:
        static log4cxx::LoggerPtr logger_;

    };
}

#endif
