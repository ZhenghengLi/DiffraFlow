#ifndef CmbConfig_H
#define CmbConfig_H

#include "GenericConfiguration.hh"

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
        string listen_host;
        int    listen_port;

    };
}

#endif
