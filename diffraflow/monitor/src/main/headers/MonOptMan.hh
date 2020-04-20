#ifndef MonOptMan_H
#define MonOptMan_H

#include "OptionsManager.hh"
#include <getopt.h>

namespace diffraflow {
    class MonOptMan: public OptionsManager {
    public:
        // option varables
        string config_file;       // -c, --config=FILE
        string logconf_file;      // -l, --logconf=FILE

    public:
        MonOptMan();
        ~MonOptMan();

        bool parse(int argc, char** argv);

    protected:
        void print_help_();

    private:
        static const char opt_string_[];
        static const option long_opts_[];

    };

}

#endif