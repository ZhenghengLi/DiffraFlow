#ifndef CmbOptMan_H
#define CmbOptMan_H

#include "OptionsManager.hh"
#include <getopt.h>

namespace diffraflow {
    class CmbOptMan: public OptionsManager {
    public:
        // option varables
        string config_file;       // -c, --config=FILE
        string logconf_file;      // -l, --logconf=FILE

    public:
        CmbOptMan();
        ~CmbOptMan();

        bool parse(int argc, char** argv);

    protected:
        void print_help_();

    private:
        static const char opt_string_[];
        static const option long_opts_[];

    };

}

#endif