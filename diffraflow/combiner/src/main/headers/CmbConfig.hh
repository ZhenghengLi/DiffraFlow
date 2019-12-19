#ifndef CmbConfig_H
#define CmbConfig_H

#include "GenericConfiguration.hh"

#include <map>

using std::map;
using std::string;

namespace diffraflow {
    class CmbConfig: public GenericConfiguration {
        public:
            int port;

        public:
            CmbConfig();
            ~CmbConfig();
            bool load(const char* filename);
            void print();
    };
}

#endif
