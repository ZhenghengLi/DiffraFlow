#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <map>

using std::map;
using std::string;

namespace diffraflow {
    class Configuration {
        public:
            int port;

        public:
        Configuration();
        ~Configuration();
        bool load(const char* filename);
        void print();
    };
}

#endif
