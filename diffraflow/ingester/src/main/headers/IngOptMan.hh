#ifndef IngOptMan_H
#define IngOptMan_H

#include "OptionsManager.hh"
#include <getopt.h>

namespace diffraflow {
    class IngOptMan: public OptionsManager {
    public:
        // option varables
        string config_file;       // -c, --config=FILE
        string logconf_file;      // -l, --logconf=FILE

    public:
        IngOptMan();
        ~IngOptMan();

        bool parse(int argc, char** argv);

    protected:
        void print_help_();

    private:
        static const char opt_string_[];
        static const option long_opts_[];

    };

}

#endif