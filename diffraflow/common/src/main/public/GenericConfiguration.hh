#ifndef GenericConfiguration_H
#define GenericConfiguration_H

#include <vector>
#include <algorithm>
#include <string>

using std::vector;
using std::pair;
using std::string;

namespace diffraflow {
    class GenericConfiguration {
    public:
        GenericConfiguration();
        ~GenericConfiguration();

        virtual bool load(const char* filename) = 0;
        virtual void print() = 0;

    protected:
        bool read_conf_KV_vec_(const char* filename,
            vector< pair<string, string> >& conf_KV_vec);

    };
}

#endif