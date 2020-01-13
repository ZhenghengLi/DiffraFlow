#ifndef DspConfig_H
#define DspConfig_H

#include "GenericConfiguration.hh"

#include <string>

using std::string;

namespace diffraflow {
    class DspConfig: public GenericConfiguration {
    public:
        DspConfig();
        ~DspConfig();

        bool load(const char* filename);
        void print();

    public:
        int dispatcher_id;
        int listen_port;
        bool compress_flag;
        string combiner_address_file;

    };
}

#endif