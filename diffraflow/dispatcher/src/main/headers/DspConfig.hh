#ifndef DspConfig_H
#define DspConfig_H

#include "GenericConfiguration.hh"

namespace diffraflow {
    class DspConfig: public GenericConfiguration {
    public:
        DspConfig();
        ~DspConfig();

        bool load(const char* filename);
        void print();

    public:
        int port;

    };
}

#endif