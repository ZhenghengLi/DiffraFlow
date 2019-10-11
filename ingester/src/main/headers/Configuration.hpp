#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <map>

using std::map;
using std::string;

namespace shine {
    class Configuration {
        public:
        map<string, string> kafka_props;
        public:
        Configuration();
        ~Configuration();
        bool load(const char* filename);
        void print();
    };
}

#endif